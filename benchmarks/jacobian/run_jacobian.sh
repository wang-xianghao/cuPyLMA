legate --cpus 16 --gpus 1 benchmarks/jacobian/run_jacobian.py \
    --slice 1000 \
    --width 128 --depth 1 \
    --mode 'forward' \
    --repeats 20

legate --cpus 16 --gpus 1 benchmarks/jacobian/run_jacobian.py \
    --slice 1000 \
    --width 128 --depth 3 \
    --mode 'reverse' \
    --repeats 20

legate --cpus 16 --gpus 1 benchmarks/jacobian/run_jacobian.py \
    --slice 1000 \
    --width 128 --depth 3 \
    --mode 'graph' \
    --repeats 20