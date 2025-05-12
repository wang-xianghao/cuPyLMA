# For running
legate --gpus 1 --cpus 16 examples/parallel_torch_curve.py --gpus 1 --slice 25000 --batch 100000 --epochs 5

# For profiling
nsys profile -o profs/parallel_torch_curve --force-overwrite true \
    --pytorch=autograd-shapes-nvtx,functions-trace \
    legate --gpus 1 --cpus 16 examples/parallel_torch_curve.py --gpus 1 --slice 25000 --batch 100000 --epochs 2