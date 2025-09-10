# Train sine cardinal curve fitting

In this tutorial, we demonstrate how to train simple dense model on curve fitting problem, taking $sinc$ function as an example.

Please refer to the full example [train.py](./train.py).

## 1. Create the dataset 
We generate train data by $y=sinc(10x)$ and construct data loader of a batch size `batch_size=10000` on it. To be noted, both the data and the loader need to be pinned to avoid extra copies via CPU<sup>[1](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)</sup>.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

batch_size = 10000

x_train = torch.linspace(-1, 1, 100000, dtype=torch.float32, pin_memory=True).unsqueeze(1)
y_train = torch.sinc(10 * x_train)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
```

## 2. Construct the model
cuPyLMA requires the model to have already been trained on a GPU device. Before creating the model, we need to retrieve the GPUs available to the model component (excluding those allocated to Legate) using `get_available_gpus`. 

We construct a simple dense neural network with a single hidden layer and select one of the available GPUs as its device.

```python
from cupylma import get_available_gpus

# Retrieve available GPUs
devices = get_available_gpus()
assert len(devices) != 0

# Construct the model on GPU device
model = SimpleDense().to(devices[0])
```

## 3. Instantiate the Levenberg-Marquardt training system
`LMA` is the core class of cuPyLMA and serves as the training system that utilizes the Levenberg-Marquardt optimizer. 

Unlike first-order optimizers (e.g., Adam), which are based on the gradient of the loss, the Levenberg-Marquardt optimizer utilizes the Jacobian matrix of the residuals. The residuals between the actual and expected outputs are defined by `residual_fn`.

`model` will be replicated on each device in `devices` to accelerate the computation of the Jacobian matrix through data parallelism. As noted, `model`'s device should be included in `devices`.


```python
from cupylma import LMA

residual_fn = lambda a, b : a - b
lma = LMA(model, devices, residual_fn)
```

## 4. Train step
`lma.step` trains the model on a batch sample.

cuPyLMA consists of two components: the model component in PyTorch and the optimizer component in cuPyNumeric, with each component holding a separate set of GPUs. Typically, we prefer to allocate more GPUs to the optimizer component to host the entire Jacobian matrix. The model component does not need to store the entire Jacobian matrix; instead, it computes the Jacobian matrix slice by slice. The `slice_size` parameter specifies the size of each Jacobian matrix slice. To benefit from data parallelism, it is recommended that `slice_size` be set to at most `batch_size // num_gpus`.

We need to use `legate.timing.time` to record the time of a step because cuPyNumeric is built on Legate, where tasks are asynchronous. The `legate.timing.time` function blocks all Legate operations.

```python
from legate.timing import time

slice_size = 5000

for x, y in train_loader:
    start_time = time()
    # Train on a batch
    loss, terminated = lma.step(x, y, slice_size)
    end_time = time()
    # Calculate one step's time in seconds
    step_time = (end_time - start_time) / 1e6
```

## 5. Execute on multiple GPUs
We need `legate` to launch the training script, as the cuPyNumeric dependency is built on the Legate runtime. The number of GPUs allocated to the optimizer component is specified by the `--gpus` option, while the model component utilizes the remaining available GPUs.

```bash
# On a node with 4 GPUs, the optimizer component receives 3 GPUs, while the model component receives 1.
legate --gpus 3 train.py
```

To execute the training on clusters using PBS, such as NCI Gadi, please refer to [gadi_v100.sh](./gadi_v100.sh).


## References
[1] Guide of using pinned memory in PyTorch: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html