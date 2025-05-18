import torch
import argparse
import cuPyLMA
import numpy as np
from legate.timing import time

device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')

# Define generic model
class Model(torch.nn.Module):
    def __init__(self, width, depth):
        super(Model, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(1, width))
        layers.append(torch.nn.Tanh())
        for _ in range(depth):
            layers.append(torch.nn.Linear(width, width))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(width, 1))

        self.denses = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.denses(x)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--slice', type=int)
parser.add_argument('--width', type=int)
parser.add_argument('--depth', type=int)
parser.add_argument('--mode', type=str)
parser.add_argument('--repeats', type=int, default=10)
args = parser.parse_args()

width = args.width
depth = args.depth
slice = args.slice
mode = args.mode
repeats = args.repeats

# Configuration for Jacobian evaluation
cuPyLMA.configuration.FORCED_JACOBIAN_MODE = True
cuPyLMA.configuration.JACOBIAN_MODE = mode

# Construct test data
X = torch.linspace(-1, 1, slice, dtype=torch.float32).unsqueeze(1).to(device)
y = torch.sinc(10 * X).to(device)

# Construct model and optimizer
loss_fn = torch.nn.MSELoss()
residual_fn = lambda a, b: torch.flatten(a - b)
model = Model(width, depth).to(device)
lma = cuPyLMA.LMA(model, [device], loss_fn, residual_fn)

# Synchronize
torch.cuda.synchronize(device)

# Get time and memory
times = []
for i in range(repeats + 1):
    start = time()
    J = lma._jacobian(X, y, residual_fn)
    torch.cuda.synchronize(device)
    end = time()
    if i > 0:
        times.append((end - start) / 1e6)

print(f'#parameters: {lma.model_size}')
print(f'norm(J): {torch.linalg.norm(J):.3f}')
print(f'avg. time: {np.mean(times):.3f} s')
print(f'std. time: {np.std(times, ddof=1):.3f} s')