legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 1_000_000_000
legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 2_000_000_000
legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 4_000_000_000
legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 8_000_000_000

legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 1_000_000_000 --pin
legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 2_000_000_000 --pin
legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 4_000_000_000 --pin
legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 8_000_000_000 --pin

legate --cpus 16 --gpus 1 benchmarks/preliminary/communication/run_h2d.py 8_000_000_000
legate --cpus 16 --gpus 2 benchmarks/preliminary/communication/run_h2d.py 8_000_000_000
legate --cpus 16 --gpus 3 benchmarks/preliminary/communication/run_h2d.py 8_000_000_000
legate --cpus 16 --gpus 4 benchmarks/preliminary/communication/run_h2d.py 8_000_000_000