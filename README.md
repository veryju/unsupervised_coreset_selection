# Unsupervised Coreset Selection(UCS)

PyTorch Code for the paper:  
**"Extending Contrastive Learning to Unsupervised Coreset Selection"**  
Jeongwoo Ju, Heechul Jung, Yoonju Oh and Junmo Kim

Original code is [SVP](https://github.com/stanford-futuredata/selection-via-proxy) from Stanford.  
Based on the original code, we implemented our unsupervised coreset algorithm.

```BibTex
@article{ju2021extending,
  title={Extending Contrastive Learning to Unsupervised Coreset Selection},
  author={Ju, Jeongwoo and Jung, Heechul and Oh, Yoonju and Kim, Junmo},
  journal={arXiv preprint arXiv:2103.03574},
  year={2021}
}
```

## Installation
### Prerequisites
- Linux or macOS (Windows is in experimental support)
- Python 3.6 +
- PyTorch 0.4.1
- TorchVision 0.2.1
- CUDA 9.1

### Shell Script File Description
| File Name | Description |
|----------|-------------|
| `run_sim_core_svhn.sh` | UCS w\ SimCLR on SVHN
| `run_sim_core_qmnist.sh` | UCS w\ SimCLR on QMNIST
| `run_sim_core_cifar.sh`  | UCS w\ SimCLR on CIFAR10
| `run_moco_core_svhn.sh` | UCS w\ MoCo on SVHN
| `run_moco_core_qmnist.sh` | UCS w\ MoCo on QMNIST
| `run_moco_core_cifar.sh` | UCS w\ MoCo on CIFAR10

### Folder Description
| Folder Name | Description |
|----------|-------------|
| `loss` | coreset score for SimCLR and MoCo
| `index` | example indices for each dataset

### Training Examples
see each shell script file in main branch
For example, ./run_sim_core_svhn.sh is as follows
```{r, engine='bash', count_lines}
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

```
## Dataset
### CIFAR10
Link: https://www.cs.toronto.edu/~kriz/cifar.html

### SVHN
Download train_32x32.mat and test_32x32.mat from the web http://ufldl.stanford.edu/housenumbers/

### QMNIST
Download QMNIST dataset using torchvision

## Experimental Results
|![coreset_acc_cifar10](https://user-images.githubusercontent.com/26498918/119467121-3655fa80-bd80-11eb-90e2-363ff96f54af.png)|
|:--:| 
| *Coreset selection performance on CIFAR10* |


|![coreset_acc_svhn](https://user-images.githubusercontent.com/26498918/119467210-44a41680-bd80-11eb-9b60-eb61fe7cbd04.png)|
|:--:| 
| *Coreset selection performance on SVHN* |

|![coreset_acc_qmnist](https://user-images.githubusercontent.com/26498918/119467228-479f0700-bd80-11eb-980a-41c32dcb0d5b.png)|
|:--:| 
| *Coreset selection performance on QMNIST* |


=======
# unsupervised_coreset_selection
pytorch source code for unsupervised coreset selection

