import argparse
import cupynumeric as np
from legate.timing import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case",
    action="append",
    nargs="+",
    help="--case <dtype> <N> <M>",
)
args = parser.parse_args()
cases = args.case
repeats = 10

def bench_single(dtype, N, M):
    if dtype == 'fp32':
        dtype = np.float32
    elif dtype == 'fp64':
        dtype = np.float64
    else:
        raise NotImplementedError()

    # Prepare matrix
    A = np.random.randn(N, M).astype(dtype)

    # Benchmark
    times = []
    for i in range(repeats + 1):
        start = time()
        if M < N:
            B = A.T @ A
        else:
            B = A @ A.T
        end = time()
        if i > 0:
            elapsed = (end - start) / 1e6
            times.append(elapsed)
    print(f'dtype {dtype} N {N}, M {M}')
    print(f'\tavg. time: {np.mean(times):.3f} s')
    print(f'\tstd. time: {np.std(times, ddof=1):.3f} s')

for dtype, N, M in cases:
    bench_single(dtype, int(N), int(M))