#!/bin/bash
#PBS -q gpuvolta
#PBS -j oe
#PBS -l walltime=00:15:00,mem=120GB
#PBS -l wd
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l storage=scratch/um09
#

source ~/.bashrc
conda activate cupylma-dev

legate --gpus 1 examples/curve/train.py --epochs 20 --batch_size 10000 --slice_size 2000