#!/bin/bash
for subsize in 22500 30000 37500 45000 52500
do
    for run in 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=2 python3 -m svp.svhn active \
            --run-dir ./run/svhn/resnet18/simclr \
            --dataset svhn \
            --datasets-dir './data' \
            --arch resnet18 \
            --num-workers 4 \
            --weighted-loss False \
            --coreset-path ./index/simclr_coreset_svhn_run$run.index \
            --coreset-loss-path ./loss/simclr_coreset_svhn_run$run.loss \
            --runs $run \
            --initial-subset $subsize \
            --eval-target-at $subsize 2>&1 | tee "./log_svhn_test_simclr_coreset_resnet18_subsize"$subsize"_run$run.txt"
     done
done