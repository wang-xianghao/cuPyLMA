# Training MNIST image classification with cuPyLMA

This tutorial demonstrates how to train an image classification model on the MNIST dataset using cuPyLMA.

Please refer to the full example [train.py](train.py).

## 1. Download and create the dataloader
We first download the MNIST dataset via `torchvision.datasets.MNIST`, which automatically downloads the dataset from the internet if it is not found locally. If your cluster's computing nodes do not have internet access, please download the dataset using [download_mnist.py](download_mnist.py) before submitting the batch job.

Next, we create a data loader for the dataset to enable training on batches. The `pin_memory` option is set to `True` to avoid extra copies.

```python
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

batch_size = 32

train_dataset = torchvision.datasets.MNIST(root=".data", train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
```

## 2. Construct the model
We construct a simple convolutional network (defined in [conv.py](conv.py)) and move it to one of the available GPUs for the model component.

```python
from cupylma import get_available_gpus
from conv import SimpleConvolutional

devices = get_available_gpus()
model = SimpleConvolutional().to(devices[0])
```

## 3. Instantiate LMA training system
To instantiate the training system `LMA`, we need provide a `residual_fn` to compute the residuals (analogous to the loss function as training with the Adam). Typically, image classification tasks utilize the categorical cross-entropy (CCE) loss, and the corresponding residual expression can be calculated using the following instructions.

```python
from cupylma import LMA

def residual_fn(a, b):
    return torch.sqrt(torch.nn.functional.cross_entropy(a, b, reduction="none"))
lma = LMA(model, devices, residual_fn)
```

### Get the residual expression from the loss function
Recalling the MSE loss, we can generalize it in terms of residuals.
$$
\begin{align*}
\text{MSE} &= \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i-y_i)^2 \\
    &= \frac{1}{n} \sum_{i=1}^{n} r_i^2
\end{align*}
$$

We can also view the CCE loss as a special MSE loss and get the residual expression ($\hat{y}$: the categorical distributions, $C$: the number of classes).
$$
\begin{align*}
CCE &= \frac{1}{n} \sum_{i=1}^{n} -\log \frac{\exp(\hat{y}_{i,y_i})}{\sum_{c=1}^C \exp(\hat{y}_{i,c})}
\end{align*}
$$
Thus, we get the residual expression for the CCE loss.
$$
r_i = \sqrt{-\log \frac{\exp(\hat{y}_{i,y_i})}{\sum_{c=1}^C \exp(\hat{y}_{i,c})}}
$$

### 4. Train step
We can simply use `LMA.step()` to train on a selected batch.

```python
for x, y in train_loader:
    loss, terminated = lma.step(x, y)
```
