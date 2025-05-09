legate --gpus 1 --cpus 16 ./benchmarks/basic/solve.py --size 40000
legate --gpus 2 --cpus 16 ./benchmarks/basic/solve.py --size 40000
legate --gpus 4 --cpus 16 ./benchmarks/basic/solve.py --size 40000
legate --gpus 8 --cpus 16 ./benchmarks/basic/solve.py --size 40000