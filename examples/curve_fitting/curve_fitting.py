import torch
import cuPyLMA
import argparse
from torch.utils.data import TensorDataset, DataLoader
from legate.timing import time
from cuPyLMA import configuration

# For debugging
torch.manual_seed(0)

# Optimization flags
configuration.OPTIM_OVERLAP_H2D = True
configuration.OPTIM_OVERLAP_D2H_H2D = True

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=2, help='number of GPUs')
parser.add_argument('--batch', type=int, default=None, help='batch size')
parser.add_argument('--slice', type=int, default=None, help='slice size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
args = parser.parse_args()
gpus = args.gpus
batch_size = args.batch
slice_size = args.slice
epochs = args.epochs

# Dense model
class DNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(1, 256), torch.nn.Tanh(), torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.linear_tanh_stack(x)

# Create dataset
x_train = torch.linspace(-1, 1, 100000, dtype=torch.float32, pin_memory=True).unsqueeze(1)
y_train = torch.sinc(10 * x_train)
x_test = torch.linspace(-1, 1, 20000, dtype=torch.float32).unsqueeze(1)
y_test = torch.sinc(10 * x_test)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
train_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

# Prepare GPU devices
devices = []
print('Model end is running on:')
for i in range(torch.cuda.device_count() - gpus, torch.cuda.device_count()):
    device_name = f'cuda:{i}'
    device = torch.device(device_name)
    devices.append(device)
    print(f'\t{device}')

# Create model and optimizer
model = DNN().to(devices[0])
def residual_fn(a, b):
    return torch.flatten(a - b)
loss_fn = torch.nn.MSELoss()
lma = cuPyLMA.LMA(model, devices, loss_fn, residual_fn)

# Test
import cupynumeric as np
all_times = []
for i in range(epochs):
    start = time()
    loss, terminated = lma.step(x_train, y_train, slice_size)
    for device in devices:
        torch.cuda.synchronize(device)
    end = time()
    print(loss)
    if i > 0:
        all_times.append((end - start) / 1e6)

print(np.mean(all_times))
print(np.std(all_times, ddof=1))