#!/usr/bin/env sh

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=AD --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4 --job-name=evalinvFusion --kill-on-bad-exit=1 python main.py --evaluate ../model_best.pth.tar -i d   2>&1|tee log/train-$now.log &
