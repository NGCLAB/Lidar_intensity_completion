#!/usr/bin/env sh

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --partition=3DV-SLAM --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4 --job-name=invFusion --kill-on-bad-exit=1 python main.py --train-mode dense -i d -b 8 --epochs 750 --lr 1e-4 --wi 0.02 --wnorm 0.02 --weight-decay 0.1 2>&1|tee log/train-$now.log &

now=$(date +"%Y%m%d_%H%M%S")
srun --partition=3DV-SLAM --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4 --job-name=invFusion --kill-on-bad-exit=1 python main.py --train-mode dense -i d -b 8 --epochs 750 --lr 1e-5 --wi 0.2 --wnorm 0.02 --weight-decay 0.1 2>&1|tee log/train-$now.log &

now=$(date +"%Y%m%d_%H%M%S")
srun --partition=3DV-SLAM --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=4 --job-name=invFusion --kill-on-bad-exit=1 python main.py --train-mode dense -i d -b 8 --epochs 750 --lr 1e-6 --wi 0.02 --wnorm 0.02 --weight-decay 0.1 2>&1|tee log/train-$now.log &
