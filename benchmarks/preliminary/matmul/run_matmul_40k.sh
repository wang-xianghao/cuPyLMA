unset CUPYNUMERIC_FAST_MATH

# Square Jacobian matrix
# 1 GPU
echo 
echo '1 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 1 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000
legate --cpus 16 --gpus 1 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000 \
    --case fp64 40_000 40_000

# 2 GPU
echo 
echo '2 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 2 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000
legate --cpus 16 --gpus 2 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000 \
    --case fp64 40_000 40_000

# 4 GPU
echo 
echo '4 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 4 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000
legate --cpus 16 --gpus 4 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000 \
    --case fp64 40_000 40_000


# 8 GPU
echo 
echo '8 GPU'
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 8 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000
legate --cpus 16 --gpus 8 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 40_000 40_000 \
    --case fp64 40_000 40_000
