import cupylma
import torch
import numpy as np
import matplotlib.pyplot as plt

from legate.timing import time
from cupylma import LMA, get_available_gpus
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from dense import SimpleDense

torch.manual_seed(0)

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
parser.add_argument('--slice_size', type=int, default=None, help='slice size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
args = parser.parse_args()
batch_size = args.batch_size
slice_size = args.slice_size
num_epochs = args.epochs

# Allocate GPUs to the model component
devices = get_available_gpus()
assert len(devices) != 0

# Create dateset
x_train = torch.linspace(-1, 1, 100000, dtype=torch.float32, pin_memory=True).unsqueeze(1)
y_train = torch.sinc(10 * x_train)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Create model on GPU
model = SimpleDense().to(devices[0])

# Create LMA optimizer
residual_fn = lambda a, b : a - b
lma = LMA(model, devices, residual_fn)

# Train
epoch_times = []
epoch_losses = []
for epoch in range(1, num_epochs + 1):
    avg_loss = 0.0
    epoch_start = time()
    for x, y in train_loader:
        loss, terminated = lma.step(x, y, slice_size)
        avg_loss += loss * y.shape[0]
    epoch_end = time()
    epoch_time = (epoch_end - epoch_start) / 1e6

    avg_loss /= len(train_dataset)
    epoch_times.append(epoch_time)
    epoch_losses.append(avg_loss)
    print(f'Epoch {epoch:3d}/{num_epochs:3d}: loss {avg_loss:10.3e}, epoch time {epoch_time:6.3f} seconds')

# Print statistics
epoch_times = epoch_times[1:-1]
print('')
print(f'Avg. epoch time: {np.mean(epoch_times):6.3f} seconds')
print(f'Std. epoch time: {np.std(epoch_times, ddof=1):6.3f} seconds')