# LiDAR Intensity Completion:
Code developed based on [self-supervised-depth-completion](https://github.com/fangchangma/self-supervised-depth-completion).

## Requirements
This code was tested with Python 3 and PyTorch 1.0 on Ubuntu 16.04.
- Install [PyTorch](https://pytorch.org/get-started/locally/) on a machine with CUDA GPU. 

## Trained Models
Download our trained models at [model_path](https://drive.google.com/file/d/1Wzus8MauSgjOOayM5VptJ-0_658ZYLz8/view?usp=sharing) to a folder of your choice.

## Training and testing
A complete list of training options is available with 
```bash
python main.py -h
```
For instance,
```bash
python main.py --train-mode dense -b 1 # train with the semi-dense annotations and batch size 1
python main.py --resume [checkpoint-path] # resume previous training
python main.py --evaluate [checkpoint-path] # test the trained model
```
