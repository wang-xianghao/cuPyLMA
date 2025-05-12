# For running
legate --gpus 1 --cpus 16 examples/curve_fitting.py --gpus 1 --slice 25000 --batch 100000 --epochs 15

# For profiling
nsys profile -o profs/curve_fitting --force-overwrite true \
    --pytorch=autograd-shapes-nvtx,functions-trace \
    legate --gpus 1 --cpus 16 examples/curve_fitting.py --gpus 1 --slice 20000 --batch 100000 --epochs 3