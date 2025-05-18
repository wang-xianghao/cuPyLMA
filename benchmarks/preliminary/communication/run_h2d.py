import torch
import argparse
import cupynumeric as np
from legate.timing import time

parser = argparse.ArgumentParser()
parser.add_argument('bytes', type=int)
args = parser.parse_args()

nbytes = args.bytes

h_data = torch.randn(nbytes // 4, dtype=torch.float32, pin_memory=True)

start = time()
d_data = np.array(h_data, dtype=np.float32)
end = time()
elapsed = (end - start) / 1e6

print(f'size: {(nbytes / 1e9):.3f} GB')
print(f'\ttime: {elapsed:.3f} s')
bandwidth = nbytes / elapsed / 1e9
print(f'\tband: {bandwidth:.3f} GB/s')