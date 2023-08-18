#!/bin/sh

for i in `seq 1 10`
do
    nohup python -u ./train_ihdp.py 4 ${i} > log_training_ihdp/output_m4_replicate${i}.log &
done
