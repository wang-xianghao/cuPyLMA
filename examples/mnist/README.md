# Training MNIST image classification with cuPyLMA

This tutorial demonstrates how to train an image classification model on the MNIST dataset using cuPyLMA.

Please refer to the full example [train.py](train.py).

## 1. Download and create the dataloader
We first download the MNIST dataset via `torchvision.datasets.MNIST`, which automatically downloads the dataset from the internet if it is not found locally. If your cluster's computing nodes do not have internet access, please download the dataset using [download_mnist.py](download_mnist.py) before submitting the batch job.

Next, we create a data loader for the dataset to enable training on batches. The `pin_memory` option is set to `True` to avoid extra CPU-side copies when transferring batches to the GPU<sup>[1](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)</sup>.

```python
train_dataset = torchvision.datasets.MNIST(root=".data", train=True, transform=ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
```

# TODO: more

## References
[1] Guide of using pinned memory in PyTorch: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html