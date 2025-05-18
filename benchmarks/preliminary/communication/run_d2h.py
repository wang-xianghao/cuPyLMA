import torch
import argparse
import time

device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')

parser = argparse.ArgumentParser()
parser.add_argument('bytes', type=int)
parser.add_argument('--pin', action='store_true')
args = parser.parse_args()

nbytes = args.bytes
pin = args.pin

d_data = torch.rand(nbytes // 4, dtype=torch.float32).to(device)
h_data = torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=pin)
torch.cuda.synchronize(device)

start = time.perf_counter()
h_data.copy_(d_data, non_blocking=pin)
torch.cuda.synchronize(device)
end = time.perf_counter()

print(f'pinned: {pin}; size: {(nbytes / 1e9):.3f} GB')
print(f'\ttime: {end - start:.3f} s')
bandwidth = nbytes / (end - start) / 1e9
print(f'\tband: {bandwidth:.3f} GB/s')