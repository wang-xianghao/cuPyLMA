cuPyLMA: a Multi-GPU Levenberg-Marquardt (Deep Learning) Optimizer Powered by NVIDIA cuPyNumeric.
=============================================

cuPyLMA is a scalable (deep learning) optimizer based on Levenberg-Marquardt algoritm. It supports multi-GPU execution via [NVIDIA cuPyNumeric](https://github.com/nv-legate/cupynumeric), which is a NumPy-like scientific computing framework.

cuPyLMA exploits the performance of multiple GPUs. cuPyLMA explicitly stores the full Jacobian matrix required by Levenberg-Marquardt algorithm for performance, which is in contrast to the most common solutions which implicitly represents the Jacobian matrix via Jacobian-vector product (JVP) and vector-Jacobian product (VJP) and thus lacks parallelism.

cuPyLMA's design consists of two components and each one holds a seperate set of GPUs.
- **Model component** hosts a PyTorch deep learning model with its data-parallelism replicas on each GPU and computes the Jacobian matrix.
- **Optimizer component** receives the Jacobian matrix from the model component and solves the optimal parameter updates by the Levenberg-Marqurdt algorithm via cuPyNumeric.

## Installation

TODO: upload to pip

## Usage
The following codes show steps to adapt exisitng PyTorch training code to utilize cuPyLMA.
```python
import cuPyLMA
import torch

class MyModel(torch.nn.Module): 
    # Implementation
model = MyModel() # Instantiate the deep learning model

# Configure optimizer
devices = [torch.device('cuda:2'), torch.device('cuda:3')] # Cuda devices held by the model component
loss_fn = torch.nn.MSELoss() # Loss function
residual_fn = lambda a, b : torch.flatten(a - b) # Residual function: the output should be an 1-d array
lma = cuPyLMA.LMA(
    model, devices,
    loss_fn, residual_fn
)

# Train one step
x_train, y_train = # train data
slice_size = # Jaocbian slice size.
             # The Jacobian matrix is decomposed into row slices for reducing the peak memory.
             # It is recommended to start from `<batch size> / <#GPUs in the model component>`.
             # If out of memory, it should be set to a smaller one.
loss, terminated = lma.step(x_train, y_train, slice_size)
```

## Performance
cuPyLMA automatically selects the best strategy for Jacobian matrix computation to reduce the peak memory usage and boost the performance.

![](./figures/jacobian_scale_batch_time_dnn.svg)
![](./figures/jacobian_scale_batch_mem_dnn.svg)

## Changelog
### Release 0.1
* First release

## Citation
In construction ...

