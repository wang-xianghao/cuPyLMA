import cupynumeric as np
import argparse
from legate.timing import time

# Parse arguments
parser = argparse.ArgumentParser(description="Solve a linear system.")
parser.add_argument('--size', type=int, default=10000)
parser.add_argument('--repeats', type=int, default=5)
args = parser.parse_args()
size = args.size
repeats = args.repeats

all_times = []

# Do experiments
for i in range(repeats + 1):
    A = np.random.rand(size, size).astype(np.float32)
    x = np.random.rand(size).astype(np.float32)
    b = A @ x

    time_start = time()
    x_solve = np.linalg.solve(A, b)
    time_end = time()

    # Exclude the first as warmup
    if i > 0:
        all_times.append((time_end - time_start) / 1e6)

avg_time = np.mean(all_times)
std_time = np.std(all_times, ddof=1)
print(f'avg. time: {avg_time:3f}')
print(f'std. time: {std_time:3f}')