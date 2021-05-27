#!/bin/bash
for subsize in 18000 24000 30000 36000 42000
do
    for run in 1 2 3 4 5 
    do
        CUDA_VISIBLE_DEVICES=0 python3 -m svp.mnist active \
            --run-dir ./run/qmnist/resnet18/moco \
            --dataset qmnist \
            --datasets-dir './data' \
            --arch resnet18 \
            --num-workers 4 \
            --weighted-loss False \
            --coreset-path ./index/moco_coreset_qmnist_run$run.index \
            --coreset-loss-path ./loss/moco_coreset_qmnist_run$run.loss \
            --runs $run \
            --initial-subset $subsize \
            --eval-target-at $subsize 2>&1 | tee "./log_qmnist_test_moco_coreset_resnet18_subsize"$subsize"_run$run.txt"
     done
done