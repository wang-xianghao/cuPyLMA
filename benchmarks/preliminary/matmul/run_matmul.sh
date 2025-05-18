unset CUPYNUMERIC_FAST_MATH

# Square Jacobian matrix
# 1 GPU
CUPYNUMERIC_FAST_MATH=1 legate --cpus 16 --gpus 1 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 20_000 20_000

legate --cpus 16 --gpus 1 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 20_000 20_000 \
    --case fp64 20_000 20_000

legate --cpus 16 --gpus 1 benchmarks/preliminary/matmul/run_matmul.py \
    --case fp32 10_000 10_000 \
    --case fp64 10_000 10_000

legate --cpus 16 --gpus 1 benchmarks/preliminary/matmul/run_matmul.py \
    fp64 \
    --case 10_000 10_000 \
    --case 20_000 20_000