import os
import sys
import torch
import logging
from lightning.fabric import Fabric

from .callbacks import EarlyStopping, ModelCheckpoint
from .history import History

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

logging.basicConfig(
    format='[%(levelname)s] : %(message)s',
    level=logging.INFO, 
    stream=sys.stdout
)
log = logging.getLogger()

torch.set_float32_matmul_precision('medium')

# To debug cuda runtime errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
 
class Trainer:
    """
    This class trains the model
    """
    def __init__(
        self, 
        model, 
        num_inputs=1, 
        device="cpu"
    ):
        """
        This is the initialization method

        Parameters
        ----------
        model : object
            A torch.nn.Module object
        num_inputs : int
            An integer specifying the number of inputs to the model
        device : device : [object, str]
            A string or torch.device object representing the device. E.g "cuda" or torch.device("cuda")

        """
        self.model = model
        self.num_inputs = num_inputs
        self.device_str = device if isinstance(device, (str,)) else device.type

        self.history = History()        
        self.is_compiled = False
    
    def _validate_compile(self):
        """
        This method validates the compiled arguments

        Returns
        -------
        None

        """
        if not isinstance(self.scheduler_step, (str,)):
            raise Exception("scheduler_step must be of type `str`")
        if not isinstance(self.metrics, (dict,)):
            raise Exception("metrics must be of type `dict`")
        if not isinstance(self.callbacks, (list,)):
            raise Exception("callbacks must be of type `list`")
        if not isinstance(self.precision, (str,)):
            raise Exception("precision must be of type `str`")
        if not isinstance(self.gradient_acc, (bool,)):
            raise Exception("gradient_acc must be of type `bool`")
        if not isinstance(self.gradient_acc_steps, (int,)):
            raise Exception("gradient_acc_steps must be of type `int`")

        if self.scheduler_step.lower() not in ("epoch", "batch"):
            raise Exception("Schedular must be of one of ('epoch', 'batch')")
            
        if self.callbacks:
            for callback in self.callbacks:
                if not isinstance(callback, (EarlyStopping, ModelCheckpoint)):
                    raise Exception("Callbacks must be of type: (EarlyStopping, ModelCheckpoint)")
                
        if self.precision not in ("32", "16-mixed", "bf16-mixed"):
            raise Exception(
                "Precision must be one of ('32', '16-mixed', 'bf16-mixed')"
            )
        else:
            if self.precision in ("16-mixed", "bf16-mixed"):
                if self.device_str == "cpu":
                    raise Exception("CPU device does not support Mixed Float16/ Brain-Float16 computations.")
                if self.precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
                    raise Exception("This cuda device does not support Brain-Float16 computations.")
                
    def _setup_compile(self):
        """
        This method sets up necessary attributes for compile

        Returns
        -------
        None
        
        """
        self.checkpoint = None
        self.checkpoint_name = None
        self.early_stopping = None

        # scheduler step
        self.scheduler_step = self.scheduler_step.lower()

        # callbacks
        if self.callbacks:
            for callback in self.callbacks:
                if isinstance(callback, (EarlyStopping,)):
                    self.early_stopping = callback
                elif isinstance(callback, (ModelCheckpoint,)):
                    self.checkpoint = callback
                    self.checkpoint_name = callback.name

        # history, progress bar and running var keys
        if self.metrics:
            for key in self.metrics:
                self.history[key] = []
                self.history[f"val_{key}"] = []
        self.keys = [
            key.replace("val_", "") for key in self.history if key.startswith("val")
        ]

        # setup fabric
        self.fabric = Fabric(
            accelerator=self.device_str,
            precision=self.precision,
            devices="auto",
            strategy="auto"
        )
        self.fabric.launch()

        ## here, fabric sets up the `model.forward` for precision auto-casting
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.model.train()

    def _update_history(self, mode, epoch_dict):
        """
        This method updates the history after each epoch

        Parameters
        ----------
        mode : str
            Either "train" or "val"
        epoch_dict : object
            A dictionary of epoch loss and/ or metrics

        Returns
        -------
        None
        
        """
        
        if mode == "train":
            prefix = ""
            self.history["lr"].append(self.optimizer.param_groups[0]['lr'])
            self.history["num_epochs"] +=  1
        else:
            prefix = "val_"

        if self.metrics:
            for metric_name in self.metrics:
                self.history[f"{prefix}{metric_name}"].append(epoch_dict[metric_name])

        self.history[f"{prefix}loss"].append(epoch_dict["loss"])

    def _calculate_metrics(self, preds, targets):
        """
        This method computes the merics

        Parameters
        ----------
        preds : object
            A torch tensor representing model predictions
        targets : float
            A torch tensor representing ground truth labels
            
        Returns
        -------
        object
        
        """

        result_metrics = {}
        for metric_name, metric_func in self.metrics.items():
            result_metrics[metric_name] = metric_func(preds, targets)
        return result_metrics

    def _get_iteration_details(self, mode, dataloader):
        """
        This method returns details needed for iteration

        Parameters
        ----------
        mode : str
            A string that specifies the mode, either `train` or `val`
        dataloader : object
            An iterator object

        Returns
        -------
        tuple

        """
        running_vars = {key : 0.0 for key in self.keys}
        len_dataloader = len(dataloader)

        if self.progress_bar:
            pbar = tqdm(enumerate(dataloader, start=1), total=len_dataloader, leave=True)
            self.pbar_postfix_keys = {
                key: key if mode == 'train' else f'{mode}_{key}'
                for key in self.keys
            }
            self.pbar_postfix_keys['lr'] = 'lr'
        else:
            pbar = enumerate(dataloader, start=1)
            self.pbar_postfix_keys = None
            
        return pbar, self.pbar_postfix_keys, running_vars, len_dataloader

    def _display_progress_bar(
        self, 
        epoch, 
        steps, 
        len_dataloader, 
        loss, 
        result_metrics, 
        epoch_dict
    ):
        """
        This method performs model training

        Parameters
        ----------
        epoch : int
            An integer defining the starting epoch
        steps: int
            An integer defining the running steps
        len_dataloader : int
            An integer defining the total number of steps
        loss : object
            A torch.tensor defining the loss
        result_metrics : dict
            A dictionary containing key as metric names and its respective values
        epoch_dict : dict | None
            A dictionary containing the epoch loss and metrics info

        Returns
        -------
        None

        """

        # update progress bar
        self.pbar.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")

        if steps < len_dataloader:
            # for all batches except for the last batch
            # values are averaged by batch_size)
            pbar_postfix_dict = {
                self.pbar_postfix_keys["loss"] : loss.item(),
                self.pbar_postfix_keys["lr"] : self.optimizer.param_groups[0]["lr"]
            }
            if self.metrics:
                for metric_name, metric_value in result_metrics.items():
                    pbar_postfix_dict[self.pbar_postfix_keys[metric_name]] = metric_value.item()

        else:
            # Epoch (last batch) -> thus dispaly total epoch loss & metrics
            pbar_postfix_dict = {
                self.pbar_postfix_keys["loss"] : epoch_dict["loss"],
                self.pbar_postfix_keys["lr"] : self.optimizer.param_groups[0]["lr"]
            }

            if self.metrics:
                for metric_name in result_metrics:
                    pbar_postfix_dict[self.pbar_postfix_keys[metric_name]] = epoch_dict[metric_name]

        self.pbar.set_postfix(pbar_postfix_dict)

    def _display_verbose(
        self, 
        mode, 
        epoch, 
        steps, 
        len_dataloader, 
        loss, 
        result_metrics, 
        epoch_dict
    ):
        """
        This method performs model training

        Parameters
        ----------
        mode : str
            A string that specifies the mode, either `train` or `val`
        epoch : int
            An integer defining the starting epoch
        steps: int
            An integer defining the running steps
        len_dataloader : int
            An integer defining the total number of steps
        loss : object
            A torch.tensor defining the loss
        epoch_dict : dict | None
            A dictionary containing the epoch loss and metrics info
        result_metrics : dict
            A dictionary containing key as metric names and its respective values

        Returns
        -------
        None

        """

        if steps < len_dataloader:
            # for all batches except for the last batch
            # values are averaged by batch_size)
            if epoch % self.verbose_epochs_frequency == 0:
                if self.verbose_steps_frequency != 0:
                    if steps % self.verbose_steps_frequency == 0:
                        verbose_mode = "" if mode == "train" else f"{mode}_"
                        verbose_list = [
                            f'Epoch [{epoch + 1}/{self.num_epochs}] | Step [{steps}/{len_dataloader}]',
                            f'{verbose_mode}Loss={loss.item()}',
                            f'lr={self.optimizer.param_groups[0]["lr"]}'
                        ]
                        if self.metrics:
                            for metric_name, metric_value in result_metrics.items():
                                verbose_list.append(f'{verbose_mode}{metric_name}={metric_value.item()}')

                        log.info(" ".join(verbose_list))

        else:
            # Epoch (last batch) -> thus dispaly total epoch loss & metrics
            if epoch % self.verbose_epochs_frequency == 0:
                verbose_mode = "" if mode == "train" else f"{mode}_"
                verbose_list = [
                    f'Epoch [{epoch + 1}/{self.num_epochs}]',
                    f'{verbose_mode}Loss={epoch_dict["loss"]}',
                    f'lr={self.optimizer.param_groups[0]["lr"]}'
                ]
                if self.metrics:
                    for metric_name in result_metrics:
                        verbose_list.append(f'{verbose_mode}{metric_name}={epoch_dict[metric_name]}')

                log.info(" ".join(verbose_list))

    def _train_loop(self, epoch, dataloader, total_len):
        """
        This method performs model training

        Parameters
        ----------
        epoch : int
            An integer defining the starting epoch
        dataloader : object
            An iterator object
        total_len : int
            An integer defining the total number of data samples

        Returns
        -------
        float

        """
        
        mode = "train"
        self.model.train()
        self.pbar, self.pbar_postfix_keys, running_vars, len_dataloader = self._get_iteration_details(mode, dataloader)

        for steps, data in self.pbar:
            batch_size = data[0].size(0)
            inputs =  data[:self.num_inputs]
            targets = data[self.num_inputs:]
            if len(targets) == 1:
                targets = targets[0]
            
            # forward
            preds = self.model(*inputs)
            loss = self.criterion(preds, targets)
                
            # backward
            if self.gradient_acc:
                is_accumulating = steps % self.gradient_acc_steps != 0
                if steps == len_dataloader:
                    is_accumulating = False

                # fabric.backward() accumulates when optimizer.zero_grad() wasn't called before it
                if steps > (len_dataloader - self.final_gradient_acc_steps):
                    self.fabric.backward(loss/ self.final_gradient_acc_steps)
                else:
                    self.fabric.backward(loss/ self.gradient_acc_steps)

                if not is_accumulating:
                    # step the optimizer and scheduler after the accumulation phase is over
                    self.optimizer.step()
                    if self.scheduler and self.scheduler_step == "batch":
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                self.fabric.backward(loss)
                # step the optimizer and scheduler
                self.optimizer.step()
                if self.scheduler and self.scheduler_step == "batch":
                    self.scheduler.step()
                self.optimizer.zero_grad()

            # update running vars
            running_vars["loss"] += loss.item() * batch_size

            # calculate metric
            if self.metrics:
                result_metrics = self._calculate_metrics(preds, targets)
                for metric_name, metric_value in result_metrics.items():
                    running_vars[metric_name] += metric_value.item() * batch_size
            else:
                result_metrics = None

            # calculate total epoch loss & metrics
            if steps < len_dataloader:
                epoch_dict = None
            else:
                # Epoch (last batch)
                epoch_dict = {"loss" : running_vars["loss"] / total_len}
                if self.metrics:
                    for metric_name in result_metrics:
                        epoch_dict[metric_name] = running_vars[metric_name] / total_len

            # display progress bar
            if self.progress_bar:
                self._display_progress_bar(epoch, steps, len_dataloader, loss, result_metrics, epoch_dict)

            # display verbose
            if self.verbose:
                self._display_verbose(mode, epoch, steps, len_dataloader, loss, result_metrics, epoch_dict)

        if self.scheduler and self.scheduler_step == "epoch":
            self.scheduler.step()

        self._update_history(mode, epoch_dict)

        return epoch_dict["loss"]

    def _validation_loop(self, epoch, dataloader, total_len):
        """
        This method performs model validation

        Parameters
        ----------
        epoch : int
            An integer defining the starting epoch
        dataloader : object
            An iterator object
        total_len : int
            An integer defining the total number of data samples

        Returns
        -------
        float
        
        """
        mode = "val"
        self.model.eval()
        self.pbar, self.pbar_postfix_keys, running_vars, len_dataloader = self._get_iteration_details(mode, dataloader)

        for steps, data in self.pbar:
            batch_size = data[0].size(0)
            inputs =  data[:self.num_inputs]
            targets = data[self.num_inputs:]
            if len(targets) == 1:
                targets = targets[0]

            # forward
            with torch.inference_mode():
                preds = self.model(*inputs)
                loss = self.criterion(preds, targets)

            # update running vars
            running_vars["loss"] += loss.item() * batch_size

            # calculate metric
            if self.metrics:
                result_metrics = self._calculate_metrics(preds, targets)
                for metric_name, metric_value in result_metrics.items():
                    running_vars[metric_name] += metric_value.item() * batch_size
            else:
                result_metrics = None

            # calculate total epoch loss & metrics
            if steps < len_dataloader:
                epoch_dict = None
            else:
                # Epoch (last batch)
                epoch_dict = {"loss" : running_vars["loss"] / total_len}
                if self.metrics:
                    for metric_name in result_metrics:
                        epoch_dict[metric_name] = running_vars[metric_name] / total_len

            # display progress bar
            if self.progress_bar:
                self._display_progress_bar(epoch, steps, len_dataloader, loss, result_metrics, epoch_dict)

            # display verbose
            if self.verbose:
                self._display_verbose(mode, epoch, steps, len_dataloader, loss, result_metrics, epoch_dict)

        self._update_history(mode, epoch_dict)

        return epoch_dict["loss"]

    def compile(
        self,
        criterion,
        optimizer,
        scheduler=None,
        scheduler_step="epoch",
        metrics={},
        callbacks=[],
        precision="32",
        gradient_acc=False,
        gradient_acc_steps=8,
    ):
        """
        This method sets the instance attributes that are required for training

        Parameters
        ----------
        criterion : object
            A criterion defining a loss function
        optimizer : object
            An optimizer
        scheduler : object | None
            A torch.optim.lr_scheduler.OneCycleLR
        scheduler_step : str
            A string representing the scheduler step. Must be one of ("epoch", "batch")
        metrics : dict
            A dictionary containing metrics
        callbacks : list
            A list containing the callbacks like EarlyStopping and/ or ModelCheckpoint
        precision : str
            Full precision ("32"), half precision AMP ("16-mixed"), bfloat16 precision AMP ("bf16-mixed")
        gradient_acc : bool
            A boolean that decides whether to perform gradient accumulation or not
        gradient_acc_steps: int
            An integer specifying the number of steps/ iterations that the gradients to be accumulated

        Returns
        -------
        None
        
        """
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.metrics = metrics
        self.callbacks = callbacks
        self.precision = precision
        self.gradient_acc = gradient_acc
        self.gradient_acc_steps = gradient_acc_steps
        
        self._validate_compile()
        self._setup_compile()
        self.is_compiled = True

    def fit(
        self, 
        num_epochs, 
        train_dataloader, 
        val_dataloader=None,
        progress_bar=False,
        verbose=False,
        verbose_epochs_frequency=1,
        verbose_steps_frequency=0
    ):
        """
        This method fits the training data for the model for "train or "val" modes

        Parameters
        ----------
        num_epochs : int
            An integer defining the total number of epochs
        train_dataloader : object
            An iterator object that represents the training data
        val_dataloader : object | None
            An iterator object that represents the validation data
        progress_bar : bool
            A boolean that specifies to show the progress bar for training results
        verbose : bool
            A boolean that specifies to print the epoch training results
        verbose_epochs_frequency : int
            An integer defining the frequecncy of epochs for verbose
        verbose_steps_frequency : int
            An integer defining the frequecncy of steps for verbose

        Returns
        -------
        object
        
        """
        if not self.is_compiled:
            raise Exception("Please compile first!")
        if progress_bar == verbose:
            verbose = not verbose if verbose else verbose
        if verbose:
            if type(verbose_epochs_frequency) == str or verbose_epochs_frequency < 1:
                raise Exception(
                    "Verbose spochs frequency cannot be less than 1 or Make sure it is an integer value."
                )
            if type(verbose_steps_frequency) == str or verbose_steps_frequency < 0:
                raise Exception(
                    "Verbose steps frequency cannot be less than 0 or Make sure it is an integer value."
                )
            
        if self.checkpoint:
            self.checkpoint._initiate()
        if val_dataloader is None:
            self.early_stopping = total_val_len = val_loss = None
            self.history.is_validation = False
        else:
            total_val_len = len(val_dataloader.dataset)
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        total_train_len = len(train_dataloader.dataset)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        self.final_gradient_acc_steps = len(train_dataloader) % self.gradient_acc_steps

        self.epoch_start = self.history["num_epochs"]
        self.num_epochs = self.epoch_start + num_epochs

        self.progress_bar = progress_bar
        self.verbose = verbose
        self.verbose_epochs_frequency = verbose_epochs_frequency
        self.verbose_steps_frequency = verbose_steps_frequency

        try:
            for epoch in range(self.epoch_start, self.num_epochs):
                # train loop
                loss = self._train_loop(epoch, train_dataloader, total_train_len)
                checkpoint_path = f"{self.checkpoint_name}_E{epoch +1}_L{loss:.4f}.pth"

                # validation loop
                if val_dataloader:
                    val_loss = self._validation_loop(epoch, val_dataloader, total_val_len)
                    checkpoint_path = checkpoint_path[:-4] + f"_VL{val_loss:.4f}.pth"
                    if self.early_stopping:
                        self.early_stopping(val_loss)

                # checkpointing
                if self.checkpoint:
                    current_loss = val_loss if val_dataloader else loss
                    if self.early_stopping:
                        if self.early_stopping.save_current_checkpoint:
                            self.checkpoint(checkpoint_path, current_loss, self.model, self.optimizer, self.scheduler)
                    else:
                        self.checkpoint(checkpoint_path, current_loss, self.model, self.optimizer, self.scheduler)

                # earlystopping
                if self.early_stopping:
                    if self.early_stopping.early_stop:
                        break
                        
            return self.history
            
        except KeyboardInterrupt:
            print("\n")
            log.warning("\033[91m[KeyboardInterrupt]\033[0m")
            log.info(f"\033[34mTerminated Training during Epoch: {epoch + 1}\033[0m")
            log.info("\033[34mView trainer history for loss and metric(s)\033[0m")
            
            return self.history
