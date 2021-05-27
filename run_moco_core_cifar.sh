#!/bin/bash
for subsize in 15000 20000 25000 30000 35000
do
    for run in 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=3 python3 -m svp.cifar active \
            --run-dir ./run/cifar/resnet18/moco \
            --dataset cifar10 \
            --datasets-dir './data' \
            --arch resnet18 \
            --num-workers 4 \
            --weighted-loss False \
            --coreset-path ./index/moco_coreset_cifar10_run$run.index \
            --coreset-loss-path ./loss/moco_coreset_cifar10_run$run.loss \
            --runs $run \
            --initial-subset $subsize \
            --eval-target-at $subsize 2>&1 | tee "./log_cifar10_test_moco_coreset_resnet18_subsize"$subsize"_run$run.txt"
     done
done
