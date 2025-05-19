unset CUPYNUMERIC_FAST_MATH

echo 
echo '1 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 1 benchmarks/preliminary/solve/run_solve.py fp32 20_000
legate --cpus 16 --gpus 1 benchmarks/preliminary/solve/run_solve.py fp32 20_000
legate --cpus 16 --gpus 1 benchmarks/preliminary/solve/run_solve.py fp64 20_000

echo 
echo '2 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 2 benchmarks/preliminary/solve/run_solve.py fp32 20_000
legate --cpus 16 --gpus 2 benchmarks/preliminary/solve/run_solve.py fp32 20_000
legate --cpus 16 --gpus 2 benchmarks/preliminary/solve/run_solve.py fp64 20_000

echo 
echo '4 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 4 benchmarks/preliminary/solve/run_solve.py fp32 20_000
legate --cpus 16 --gpus 4 benchmarks/preliminary/solve/run_solve.py fp32 20_000
legate --cpus 16 --gpus 4 benchmarks/preliminary/solve/run_solve.py fp64 20_000
