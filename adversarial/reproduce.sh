#!/bin/bash

TOLS=(0.001 0.01 0.1 1 10 0.0001)

if [[ "$1" =~ "mnist" ]]; then
    STEP=0.05
    EPSS=(0.05 0.1 0.3)
else
    STEP=0.01
    EPSS=(0.01 0.03 0.05)
fi

if [[ "$1" =~ "resnet" ]]; then
    TOLS=(${TOLS[@]: -1:1})  # only one TOL (the smallest one for convenience)
    echo ${TOLS[@]}
fi

for EPS in ${EPSS[@]}; do
for TOL in ${TOLS[@]}; do
for P in inf 2; do

python adversarial/attack.py -t $TOL -e $EPS -d $P -s $STEP $1
python adversarial/diff.py -t $TOL -e $EPS -d $P -s $STEP $1

done
done
done
