import torch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--size', type=int, default=100000)
parser.add_argument('--block', action='store_true')
args = parser.parse_args()
ngpus = args.gpus
size = args.size
block = args.block

devices = []
As = []
Bs = []
for i in range(ngpus):
    device = torch.device(f'cuda:{i}')
    devices.append(device)
    As.append(torch.rand(size, size, device=device))

for device in devices:
    torch.cuda.synchronize(device=device)

start_time = time.perf_counter()
for i in range(ngpus):
    Bs.append(As[i] @ As[i])
    if block:
        torch.cuda.synchronize(device=devices[i])

for device in devices:
    torch.cuda.synchronize(device=device)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f'time: {elapsed_time:.3f}')