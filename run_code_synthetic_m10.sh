#!/bin/sh

for i in `seq 1 10`
do
    nohup python -u ./train_synthetic.py 10 ${i} > log_training_synthetic/output_m10_replicate${i}.log &
done
