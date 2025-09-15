import cupylma
import torch
import torchvision
import numpy as np

from legate.timing import time
from cupylma import LMA, get_available_gpus
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from argparse import ArgumentParser
from conv import SimpleConvolutional

torch.manual_seed(0)

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--slice_size', type=int, default=None, help='slice size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
args = parser.parse_args()
batch_size = args.batch_size
slice_size = args.slice_size
num_epochs = args.epochs

# Allocate GPUs to the model component
devices = get_available_gpus()
assert len(devices) != 0

# Prepare dataset
train_dataset = torchvision.datasets.MNIST(root=".data", train=True, transform=ToTensor(), download=True)
# test_dataset = torchvision.datasets.MNIST(root=".data", train=False, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create model on GPU
model = SimpleConvolutional().to(devices[0])

# Create LMA optimizer
def residual_fn(a, b):
    return torch.sqrt(torch.nn.functional.cross_entropy(a, b, reduction="none"))
lma = LMA(model, devices, residual_fn)

# Train
epoch_times = []
for epoch in range(1, num_epochs + 1):
    avg_loss = 0.0
    
    epoch_start = time()
    for x, y in train_loader:
        loss, terminated = lma.step(x, y, slice_size)
        avg_loss += loss * y.shape[0]
    epoch_end = time()
    epoch_time = (epoch_end - epoch_start) / 1e6

    avg_loss /= len(train_dataset)
    epoch_times.append(avg_loss)
    print(f'Epoch {epoch:3d}/{num_epochs:3d}: loss {avg_loss:10.3e}, epoch time {epoch_time:6.3f} seconds')

# Print statistics
epoch_times = epoch_times[1:-1]
print('')
print(f'Avg. epoch time: {np.mean(epoch_times):6.3f} seconds')
print(f'Std. epoch time: {np.std(epoch_times, ddof=1):6.3f} seconds')