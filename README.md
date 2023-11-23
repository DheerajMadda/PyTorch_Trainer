## Trainer </br>

Development of training PyTorch models is not so simple. The training scripts are not generalizable from project to project. For a project, one has to write custom training loops, keep track of loss and metric calculations, to use learning rate schedulers if necessary, and so on. Once it is done, utilizing these scripts for a different project requires a lot of code changes to be done and it is a time consuming task. </br>

This Trainer makes development of Pytorch training extremely easy and fast while making it as generic as possible from project to project. It helps an individual to focus more on data preprocessing and experimenting various model architectures rather than spending more than sufficient amount of time on writing training scripts. The Trainer also provides many interesting features which can be easily used as required. </br>

</br>
</br>

<img width="849" alt="pytorch_board_v2" src="https://github.com/DheerajMadda/test/assets/50489165/46b50ce8-5d13-4e18-b639-de49f173536c">

</br>
</br>

The Trainer features the following: </br>

- Supports **Model Profiling**: Model Size, Num_Parameters, MACs, FLOPs, Inference Latency </br>

- Supports **Model Types**: Single-Input-Single-Output, Single-Input-Multiple-Outputs, Multiple-Inputs-Multiple-Outputs </br>

- Supports **Learning Rate Scheduler**: </br>

- Supports **Metrics**: Single or multiple metric(s) </br>

- Supports **Callbacks**: EarlyStopping, ModelCheckpoint </br>

- Supports **Training Precisions**: Single Precision - FP32, Mixed-Precisions - FP16 AMP, BF16 AMP </br>

- Supports **Gradient Accumulation**: Accumulates gradients over several batches </br>

- Supports **Training History**: Epoch, learning rate, loss and/ or metric(s) </br>

- Supports **Training Progress Bar**: Epoch, learning rate, loss and/ or metric(s) </br>

</br>
</br>

Note: </br>

- By default, the Trainer trains the model with FP32 precision with given criterion and optimizer. All other features can be used as required. </br>

- It is recommended and important that the loss and metric functions must return an averaged value or reduction by 'mean'. </br>

- It is suggested to use PyTorch 2.0 and above, as the Trainer is tested for the same. </br>

- It is recommended to use [torchmetrics](https://pypi.org/project/torchmetrics/) as it supports computations on "cpu" as well as "cuda". </br>

- Not all operations or GPUs support BF16 precision. CPU does not support Mixed precision training at the moment. The trainer will raise and Exception upon trainer.fit() if there is an occurance of any such case.  </br>

- To get started with this Trainer, please go through [this](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/notebooks/1_Torch_Trainer_Tutorial.ipynb) notebook. </br>

</br>
</br>

### Experimentation: </br>
An experimentation has been carried out to compare the various training precisions using 3 factors; i) CUDA memory usage during training ii) training time iii) accuracy achieved. The notebook for this experimentation can be found [here](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/demo/Classification.ipynb). </br>

The model used is [EfficientNet-B5](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b5.html) and dataset used is [Standford Cars Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset). For a fair comparison, the model is only trained for 10 epochs and with same settings for various training precisions on device: Nvidia RTX 3080 TI 16GB. </br>

EfficientNet-B5 details: </br>

- Model size = 110.542 MB </br>

- Number of parameters = ~28.74 Million </br>

- MACs = ~2.458 GMACs </br>

- FLOPs = ~4.916 GFLOPs

Note: MACs and FLOPs are with respect to batch size of 1. </br>

Please find the following experimentation result that is carried out for 10 epochs: </br>

<img width="960" alt="experiments" src="https://github.com/DheerajMadda/test/assets/50489165/fc7dcb80-5a52-4c11-b04b-f5ddb7d059d8">

</br>

Note: When using FP16 AMP + Gradient Accumulation, the accuracy achieved is less compared to FP32 training. To get to the same level of accuracy as of baseline FP32, it requires to be trained for more number of epochs as compared to baseline FP32 training. </br>

After training the model (FP16 AMP + Gradient Accumulation with batch size=6, gradient accumulation steps=8) for an additional 6 epochs, following displays the additional time required to reach to the accuracy comparable to baseline FP32. </br>

<img width="848" alt="additional_time" src="https://github.com/DheerajMadda/test/assets/50489165/4abdaf25-e4b4-4030-a3a7-a4b6ecdaee99">

</br>

Information obtained through this experimentation:- </br>

- Mixed Precision trainings practically are very efficient and faster compared to Single Precision (FP32) training. They reach to the same level of accuracy compared to the baseline FP32 training within the same number of epochs. </br>

- There is not much of a difference between 16-bit and bf16-bit mixed precision trainings when considering factors like cuda memory usage, time required to train and achieving same level of accuracy. </br>

- Gradient Accumulation is an amazing technique to reduce cuda memory usage, but comes at an expense of additional time as it requires more number of epochs to reach to the expected accuracy compared to the baseline FP32. Can we reduce number of epochs? -> Yes, try using a higher learning rate than normal.

- In Gradient Accumulation, as we go on decreasing the batch size or increasing the gradient accumulation steps to get to the desired virtual batch size, the training time increases. </br>

</br>
</br>

### Usage Examples: </br>

<img width="580" alt="profiling" src="https://github.com/DheerajMadda/test/assets/50489165/62bf0e7f-7d5b-4f47-8e8e-32396d4beedc">

</br>
</br>

<img width="512" alt="trainer" src="https://github.com/DheerajMadda/test/assets/50489165/ac3d4570-a6a0-46d9-a266-a647371537a3">
<img width="720" alt="progress_bar" src="https://github.com/DheerajMadda/test/assets/50489165/af4370fa-d568-4d08-b0d3-1a5dc8094e32">

</br>
</br>
</br>

<img width="850" alt="history" src="https://github.com/DheerajMadda/test/assets/50489165/1743f4b9-9be7-4fa7-a81f-d562ad784534">

</br>
</br>

#### Directory and files information:-
- [notebooks](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/notebooks/) -> contains the jupyter notebook. 

- [torch_trainer](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/torch_trainer) -> contains the profiling and training utilities.

- [utils](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/utils) -> contains the loss and metrics definition.

- [demo](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/demo) -> contains the experimentation notebook.

- [requirements.txt](https://github.com/DheerajMadda/PyTorch_Trainer/blob/main/requirements.txt) -> contains all the required libraries to be installed.

</br>
</br>

<hr>

### Please read the following to understand the features provided by the Trainer: </br>

- **Model Profiling:** It is always good to perform the model profiling before training to get the complexity of the model. It computes the Model Size, Num_Parameters, MACs, FLOPs, Inference Latency. It supports profiling for both devices, "cpu" and "cuda". </br>

- **Model Types:** </br>

&emsp;&emsp; □ **Single-Input-Single-Output:** The Trainer can train a model that accepts single input and produces a single output, a torch.tensor(). </br>

&emsp;&emsp; □ **Single-Input-Multiple-Outputs:** The Trainer can train a model that accepts single input and produces multiple outputs, i.e tuple(torch.tensor(), torch.tensor(), ..., torch.tensor()). </br>

&emsp;&emsp; □ **Multiple-Inputs-Multiple-Outputs:** The Trainer can train a model that accepts multiple inputs, i.e. torch.tensor(), torch.tensor(), ..., torch.tensor() and produces multiple outputs, i.e tuple(torch.tensor(), torch.tensor(), ..., torch.tensor()).</br>

- **Learning Rate Scheduler:** The Trainer only supports OneCycleLR scheduler. It is a widely used scheduler and unlike StepLR/ MultiStepLR and many other schedulers, it updates the optimizer's learning rate over each batch. It is based on a 2018 paper titled "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (https://arxiv.org/abs/1708.07120) </br>

- **Metrics:** The trainer supports single or multiple metric(s) by defining it as a dictionary, where key is the metric name and its value should be the metric function. </br>

- **Callbacks:** </br>

&emsp;&emsp; □ **EarlyStopping:** It uses a patience (an integer value) that determines the number of times to wait after last time validation loss improved before stopping the training. It only works if validation dataloader is used! </br>

&emsp;&emsp; □ **ModelCheckpoint:** It saves the checkpoint(s) (model, optimizer, scheduler) to the disk for each epoch. It also features to only save the best checkpoint. </br>

- **Training Precisions:** </br>

&emsp;&emsp; □ **FP32**: This is the default training precision (single-precision) of the Trainer. Range:- 1.17e-38 to 3.40e38 </br>

&emsp;&emsp; □ **FP16 AMP (Automatic Mixed Precision)**: FP16 Range:- -65504 to 65504. FP16 AMP trains the model in both FP32 and FP16. The reduction in memory consumption may not be significant. It is preferred over true FP16 as it uses both the single and half precisions that will avoid producing NaN values during training. There will be significant reduction in memory usage. </br>

&emsp;&emsp; □ **BF16 AMP (Automatic Mixed Precision)**: Brain-Floating, BFP16 (half-precision: a format that was developed by Google Brain, an artificial intelligence research group at Google). It helps in reducing memory consumption. It has the same dynamic range as FP32. It is important to note that it is only supported on the Ampere architecture GPUs and the Trainer will raise an Exception if it is compiled with BF16 for CPU, or the GPU that does not support it. Range:- 1.17e-38 to 3.40e38. Now, BF16 AMP (Automatic Mixed Precision), it trains the model in both FP32 and BFP16. There will be significant reduction in memory usage.  </br>

- **Gradient Accumulation:** The Trainer comes with easy to use gradient accumulation technique.To use the gradient accumulation, **gradient_acc_steps** need to be set as an integer value that specifies the number of steps the gradients should be accumulated before updating the model parameters. </br>

- **Training History:** It records the training history:- epoch, learning rate, loss and/or metric(s). It also provides methods to plot the loss and metric(s). </br>

- **Training Progress Bar:** It shows progress bar for training while displaying epoch, learning rate, loss and/or metric(s) for each iteration as well as for an epoch.

</br>
</br>

### What is Mixed-Precision Training?
- Mixed precision training uses both 16-bit and 32-bit precision to ensure no loss in accuracy. The computation of gradients in the 16-bit representation is much faster than in the 32-bit format and saves a significant amount of memory. This strategy is beneficial, especially when we are memory or compute-constrained. </br>

- It’s called “mixed-“rather than “low-“precision training because we don’t transfer all parameters and operations to 16-bit floats. Instead, we switch between 32-bit and 16-bit operations during training, hence, the term “mixed” precision. </br>

Following is an example of FP16-bit mixed precision training. Note that, BF16-bit mixed precision training is also similar. </br>

<img width="512" alt="mixed_precision" src="https://github.com/DheerajMadda/test/assets/50489165/21ddbb5c-5bf6-4059-87f4-9d1ca44bd7d8">

</br>
</br>

### What is Gradient Accumulation?
- Gradient accumulation is a way to virtually increase the batch size during training, which is very useful when the available GPU memory is insufficient to accommodate the desired batch size. Note that this only affects the runtime, not the modeling performance. </br>

- In gradient accumulation, gradients are computed for smaller batches and accumulated (usually summed or averaged) over multiple iterations instead of updating the model weights after every batch. Once the accumulated gradients reach the target “virtual” batch size, the model weights are updated with the accumulated gradients. </br>

- For example, if we want to use a batch size of 256 but can only fit a batch size of 64 into GPU memory, we can perform gradient accumulation over four batches of size 64. (After processing all four batches, we will have the accumulated gradients equivalent to a single batch of size 256.) This allows us to effectively emulate a larger batch size without requiring larger GPU memory or tensor sharding across different devices. </br>

- While gradient accumulation can help us train models with larger batch sizes, it does not reduce the total computation required. In fact, it can sometimes lead to a slightly slower training process, as the weight updates are performed less frequently. Nevertheless, it allows us to work around limitations where we have very small batch sizes that lead to noisy updates.

In short:- </br>

Instead of using an actual batch size of 16 and thus updating the model parameters for an effective batch size of 16, using gradient accumulation with 4 accumulation steps means we will now use an actual batch size of 4 (since 16 / 4 = 4). Now we are reducing the batch size to 4 and thus the memory consumption, but we are updating the model parameters for an effective batch size of 16 only. <br>

</br>
</br>

### (Optional) Why not to train models with just Half Precisions - FP16 or BF16?
While training with just half precisions like FP16 anf BF16 is possible, it is often not recommended! </br>

<img width="512" alt="diff_precisions" src="https://github.com/DheerajMadda/test/assets/50489165/d40126fc-f2f0-4915-95f9-5b01a13cccdd">

FP32:- Dynamic range = 1.17e-38 to 3.40e38; Precision = 6–9 significant decimals. </br>

FP16:- Dynamic range = -65504 to 65504; Precision = 3-4 significant decimals.</br>

BF16:- Dynamic range = 1.17e-38 to 3.40e38; Precision = 2–3 significant decimals. </br>

If you train a model with FP16-bit precision, you may encounter NaN values in the loss: </br>

Epoch: 01/100 | Batch 000/703 | Loss: 2.4105 </br>

Epoch: 01/100 | Batch 300/703 | Loss: nan </br>

Epoch: 01/100 | Batch 600/703 | Loss: nan </br>

Because, regular 16-bit floats can only represent numbers between -65,504 and 65,504. </br>

Now the extended dynamic range helps bf16 to represent very large and very small numbers, making it more suitable for deep learning applications where a wide range of values might be encountered. However, it has even lower decimal precision than the regular 16-bit and this may affect the accuracy of certain calculations or lead to rounding errors in some cases. This may affect the the performance of the model. Thus, regular FP16 or BF16 is not preferred for training the models, but instead its mixed versions (FP16-FP32, BF16-FP32) are preferred and are widely used. </br>

</br>
</br>

### (Optional) What are FLOPs and MACs?
Complexity of the model can be measured using the model size, Floating Point Operations(FLOPs), Multiply-Accumulate Computations(MACs), the number of model parameters, and the inference latency. These are model KPIs. </br>

Let us understand what are FLOPs and MACs. </br>

- FLOPs:- </br>

We can determine the total amount of calculations the model will need to carry out in order to estimate the inference time for that model. FLoating point OPeration, or FLOP, actually is an operation involving a floating point value, including addition, subtraction, division, and multiplication falls under this category. </br>

- MACs:- </br>

Multiply-Accumulate Computations, or MACs. MAC is an operation that performs two operations - addition and multiplication. A neural network always performs additions and multiplications. (e.g input * weight + bias). We typically assume that 1 MAC = 2 FLOPs. </br>

- (Optional) FLOPS:- </br>

&emsp;&emsp; □ The FLOPS, with a capital S; It is not a model KPI. FLoating point OPerations per Second or FLOPS, is a rate that provides information about the quality of the hardware on which the model is supposed to be deployed. The inference will happen more quickly if the more operations per second we can perform. </br>

&emsp;&emsp; □ Estimating Inference Time even before training or building a model:- </br>

Consider a neural network that has 'x' Parameters (e.g. weights and biases). </br>

Layer 1 = w1∗p1 + w2∗p2 + ... + wn∗pn = 1000 FLOPs and so on for each layer. </br>

Let's say it requires a total of 1,000,000 (1M) FLOPs to produce an output. </br>

Consider a CPU with a 1 GFLOPS performance. Then the inference time = FLOPs/FLOPS = (1,000,000)/(1,000,000,000) = 1ms. </br>

General rule:- </br>

- The model should have few FLOPs while still being sufficiently complex to be useful. </br>

- The hardware should have a lot of FLOPS. </br>

</br>
</br>
