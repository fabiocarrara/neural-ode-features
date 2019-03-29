#/bin/bash

LR=0.1
BS=128
DROPOUT=0.5
WD=1e-4

## MNIST
COMMON="--filters 64 --dataset mnist --augmentation none --epochs 25 --batch-size $BS --lr $LR --patience 5 --dropout $DROPOUT --wd $WD"

python train.py --model resnet --downsample residual $COMMON
python train.py --model odenet --downsample residual $COMMON
python train.py --model odenet --downsample one-shot $COMMON

## CIFAR-10
COMMON="--filters 256 --dataset cifar10 --augmentation crop+jitter+flip+norm --epochs 250 --batch-size $BS --lr $LR --patience 15 --dropout $DROPOUT --wd $WD"

python train.py --model resnet --downsample residual $COMMON
python train.py --model odenet --downsample residual $COMMON
python train.py --model odenet --downsample one-shot $COMMON