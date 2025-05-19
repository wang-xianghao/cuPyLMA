import argparse
import cupynumeric as np
import numpy
import csv

parser = argparse.ArgumentParser()
parser.add_argument('out', type=str)
parser.add_argument('dtype', type=str)
parser.add_argument('size', type=int)
args = parser.parse_args()
out = args.out
dtype = args.dtype
size = args.size

repeats = 1000

if dtype == 'fp32':
    dtype = np.float32
elif dtype == 'fp64':
    dtype = np.float64
else:
    raise NotImplementedError()

f = open(out, 'w')
errors = []
for i in range(repeats + 1):
    A_ = np.random.randn(size, size).astype(numpy.float64)
    x_ = np.random.randn(size, size).astype(numpy.float64)
    b_ = A_ @ x_

    A = np.array(A_, dtype=dtype)
    b = np.array(b_, dtype=dtype)

    x = numpy.linalg.solve(A, b)
    if i > 1:
        f.write(str(float(np.linalg.norm(x_ - x))) + '\n')
        # err = float(np.sum(np.power(x_ - x, 2)) / len(x_))
        # print(err)
        # f.write(f'{err}\n')

f.close()