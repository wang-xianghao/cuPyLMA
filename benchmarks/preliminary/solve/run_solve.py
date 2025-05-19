import argparse
import cupynumeric as np
from legate.timing import time

parser = argparse.ArgumentParser()
parser.add_argument('dtype', type=str)
parser.add_argument('size', type=int)
args = parser.parse_args()
dtype = args.dtype
size = args.size
repeats = 10

if dtype == 'fp32':
    dtype = np.float32
elif dtype == 'fp64':
    dtype = np.float64
else:
    raise NotImplementedError()

A = np.random.rand(size, size).astype(dtype)
x = np.random.rand(size).astype(dtype)
b = A @ x

times = []
for i in range(repeats + 1):
    start = time()
    x_expect = np.linalg.solve(A, b)
    end = time()
    elapsed = (end - start) / 1e6
    if i > 1:
        times.append(elapsed)

print(f'dtype {dtype} size {size}')
print(f'\tavg. time: {np.mean(times):.3f} s')
print(f'\tstd. time: {np.std(times, ddof=1):.3f} s')