#!/usr/bin/env sh

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=AD --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4 --job-name=invFusion --kill-on-bad-exit=1 python main.py --train-mode dense -i d -b 8 --epochs 750 --lr 1e-4 --wi 0.02 --wpure 0.3  2>&1|tee log/train-$now.log &
