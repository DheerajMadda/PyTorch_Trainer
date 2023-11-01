import os
import sys
import copy
import json
import numpy as np
import pandas as pd

import time
import logging
import tempfile
import torch
from thop import profile

logging.basicConfig(
    format='[%(levelname)s] : %(message)s',
    level=logging.INFO, 
    stream=sys.stdout
)
log = logging.getLogger()

class ProfilerResult:
    """
    This class provides a means to store the Profiling results
    """
    def __init__(self):
        """
        This is the initialization method
        """
        self._columns = ["PARAMETERS", "CPU", "CUDA"]
        self._data_df = {key : [] for key in self._columns}
        self.results = {
            "CPU" : None,
            "CUDA" : None
        }
        
    def __str__(self):
        """
        This method returns the serialized profiler result

        Returns
        -------
        str

        """
        return json.dumps(self.results, indent=4)

    def to_dict(self):
        """
        This method returns the profile result

        Returns
        -------
        dict

        """
        return self.results

    def to_pandas(self):
        """
        This method creates a pandas datafrom from the dictionary result

        Returns
        -------
        object

        """
        data_df = copy.deepcopy(self._data_df)
        columns = copy.deepcopy(self._columns)
        
        for device, data in self.results.items():
            if data:
                for row_name, row_data in data.items():
                    if len(data_df["PARAMETERS"]) < len(data):
                        data_df["PARAMETERS"].append(row_name)
                    data_df[device].append(str(row_data))
            else:
                data_df.pop(device)
                columns.remove(device)

        df = pd.DataFrame(data_df, columns=columns)
        return df

class Profiler:
    """
    This class provides methods to perform various types of profiling
    """
    KILOBYTE_TO_BYTE = MEGABYTE_TO_KILOBYTE = 1024

    def get_model_size(self, model):
        """
        This method computes the model size

        Parameters
        ----------
        model : object
            A torch.nn.Module object

        Returns
        -------
        float

        """
        size_of_parameters = sum([param.element_size() * param.nelement() for param in model.parameters()])
        size_of_buffers = sum([buf.element_size() * buf.nelement() for buf in model.buffers()])
        model_size = size_of_parameters + size_of_buffers # Bytes
        model_size_mb = model_size / (self.KILOBYTE_TO_BYTE * self.MEGABYTE_TO_KILOBYTE) # MegaBytes
        return round(model_size_mb, 3)

    def get_model_size_on_disk(self, model):
        """
        This method computes the model size on the disk

        Parameters
        ----------
        model : object
            A torch.nn.Module object

        Returns
        -------
        float
        
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_save_path = os.path.join(tmp_dir, "model.pth")
            torch.save(model.state_dict(), model_save_path)
            model_file_size_mb = os.path.getsize(model_save_path) / (self.KILOBYTE_TO_BYTE * self.MEGABYTE_TO_KILOBYTE)
        return round(model_file_size_mb, 3)

    def get_inference_time(self, model, inputs, device, n_iters=10, gpu_warmup=False):
        """
        This method computes the inference time of the model for both targets, cpu and cuda

        Parameters
        ----------
        model : object
            A torch.nn.Module object
        inputs : object
            A numpy array or torch tensor
        device : [object, str]
            A string or torch.device object representing the device. E.g "cpu" or torch.device("cpu")
        n_iters : int
            An integer value specifying the number of iterations to be performed for a given 
            input batch
        gpu_warmup : bool
            If set to True; warms up the GPU before computing the actual GPU inference time

        Returns
        -------
        float
        
        """

        is_gpu = False if str(device) == "cpu" else True
        batch_size = inputs[0].shape[0]
        model.to(device).eval()
        timings = np.zeros((n_iters, 1))

        if is_gpu:
            if gpu_warmup:
                # GPU-WARM-UP
                warm_ups = [
                    [(1, 4), 12],
                    [(5, 8), 9],
                    [(8, 12), 6],
                    [(12, 16), 3]
                ]
                
                if batch_size > 16:
                    num_warm_ups = 1
                else:
                    for warm_up in warm_ups:
                        if warm_up[0][0] <= batch_size <= warm_up[0][1]:
                            num_warm_ups = warm_up[1]
                            break

                log.info(f"Performing GPU warm up with {num_warm_ups} iterations...")
                for _ in range(num_warm_ups):
                    model(*inputs)
                log.info("GPU warmup complete!")

            with torch.inference_mode():
                for rep in range(n_iters):
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    
                    starter.record()
                    model(*inputs)
                    ender.record()
                    
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()

                    elapsed_time = starter.elapsed_time(ender)  # milliseconds
                    timings[rep] = elapsed_time / 1000 # seconds

        else:
            with torch.inference_mode():
                for rep in range(n_iters):
                    start_time = time.perf_counter()
                    model(*inputs)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time  # seconds
                    timings[rep] = elapsed_time

        avg_batch_inference_time = round(np.sum(timings)/ n_iters, 3)
        avg_sample_inference_time = avg_batch_inference_time / batch_size

        return avg_batch_inference_time, avg_sample_inference_time

    def get_model_metrics(self, model, inputs, device):
        """
        This method computes the model metrics - MACs, FLOPs and number of parameters

        Parameters
        ----------
        model : object
            A torch.nn.Module object
        inputs : object
            A numpy array or torch tensor
        device : object | str
            A string or torch.device object representing the device. E.g "cpu" or torch.device("cpu")
            
        Returns
        -------
        tuple
        
        """

        batch_size = inputs[0].shape[0]
        model.to(device).eval()
        
        MACs, params = profile(model, inputs=inputs, verbose=False)

        M_params = round(params * 1e-6, 6)
        M_MACs_batch = round(MACs * 1e-6, 6) 
        M_FLOPs_batch = 2 * M_MACs_batch

        M_MACs = M_MACs_batch / batch_size
        M_FLOPs = M_FLOPs_batch / batch_size

        return M_MACs, M_MACs_batch, M_FLOPs, M_FLOPs_batch, M_params

    def __call__(self, model, inputs, devices=["cpu"], n_iters=10, gpu_warmup=False):
        """
        This method performs all defined profiling methods for both targets, cpu and cuda

        Parameters
        ----------
        model : object
            A torch.nn.Module object
        inputs : object
            A numpy array or torch tensor
        devices : list[str]
            A list containing the device type (string) E.g ["cpu"], ["cuda"], ["cpu", "cuda"]
        n_iters : int
            An integer value specifying the number of iterations to be performed for a given 
            input batch
        gpu_warmup : bool
            If set to True; warms up the GPU before computing the actual GPU inference time

        Returns
        -------
        object
        
        """

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for idx in range(len(inputs)):
            if not isinstance(inputs[idx], (torch.Tensor,)):
                inputs[idx] = torch.from_numpy(inputs[idx])
        
        pr = ProfilerResult()
        compute_only_once = True

        start_time = time.perf_counter()
        log.info("Profiling is running...")

        for device in devices:
            device = device.lower()
            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx].to(device)

            model_size = self.get_model_size(model)
            model_size_on_disk = self.get_model_size_on_disk(model)
            avg_batch_inference_time, avg_sample_inference_time = \
                self.get_inference_time(model, inputs, device, n_iters, gpu_warmup)
            
            if compute_only_once:
                M_MACs, M_MACs_batch, M_FLOPs, M_FLOPs_batch, M_params = \
                    self.get_model_metrics(model, inputs, device)
                compute_only_once = False
            
            pr.results[device.upper()] = {
                "MODEL SIZE (MB)" : model_size,
                "MODEL SIZE ON DISK (MB)" : model_size_on_disk,
                "NUM PARAMETERS (Million)" : M_params,
                "SAMPLE MACs (Mega)" : M_MACs,
                "SAMPLE FLOPs (Mega)" : M_FLOPs,
                "SAMPLE AVERAGE INFERENCE TIME (Seconds)" : avg_sample_inference_time,
                "BATCH MACs (Mega)" : M_MACs_batch,
                "BATCH FLOPs (Mega)" : M_FLOPs_batch,
                "BATCH AVERAGE INFERENCE TIME (Seconds)" : avg_batch_inference_time,
                "INFERENCE BATCH SIZE" : inputs[0].shape[0],
                "ITERATIONS PER BATCH" : n_iters
            }
            
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        log.info("Completed!")
        log.info(f"Time taken: {elapsed_time:.3f} seconds.")

        return pr
