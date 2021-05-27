import os
import numpy as np

from typing import Optional, List, Callable, Any
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

Transform = Callable[[Any], Any]

DATASETS = [ 'mnist','fmnist', 'emnist', 'emnist_letters','qmnist']

MEANS = { 'mnist'  : (0.5),
          'fmnist' : (0.5),
          'emnist' : (0.5),
          'emnist_letters' : (0.5),
          'qmnist' : (0.5)
        }

STDS = { 'mnist'  : (0.5),
         'fmnist' : (0.5),
         'emnist' : (0.5),
         'emnist_letters' : (0.5),
         'qmnist' : (0.5) 
       }


def _create_test_dataset(dataset, dataset_dir, transform,
                         target_transform=None):
    if dataset == 'mnist':
        test_dataset = datasets.MNIST(root=dataset_dir, train=False,
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'fmnist':
        test_dataset = datasets.FashionMNIST(root=dataset_dir, train=False,
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'emnist':
        test_dataset = datasets.EMNIST(root=dataset_dir, split='digits',
                                      train=False,  
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'emnist_letters':
        test_dataset = datasets.EMNIST(root=dataset_dir, split='letters',
                                      train=False,  
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'qmnist':
        test_dataset = datasets.QMNIST(root=dataset_dir, what='test',
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    else:
        raise NotImplementedError(f'{dataset} is not an available option.')
    return test_dataset


def _create_train_dataset(dataset, dataset_dir, transform,
                          target_transform=None):
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root=dataset_dir, train=True,
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=dataset_dir, train=True,
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'emnist':
        train_dataset = datasets.EMNIST(root=dataset_dir, split='digits',
                                      train=True,  
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'emnist_letters':
        train_dataset = datasets.EMNIST(root=dataset_dir, split='letters',
                                      train=True,  
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    elif dataset == 'qmnist':
        train_dataset = datasets.QMNIST(root=dataset_dir, what='train',
                                      download=True,
                                      transform=transform,
                                      target_transform=target_transform)
    else:
        raise NotImplementedError(f'{dataset} is not an available option.')

    return train_dataset



def create_dataset(dataset: str, datasets_dir: str,
                   transform: Optional[List[Transform]] = None,
                   target_transform: Optional[List[Transform]] = None,
                   train: bool = True,
                   augmentation: bool = True) -> Dataset:
    """
    Create CIFAR datasets.

    Parameters
    ----------
    dataset: str
        Name of dataset.
    datasets_dir: str
        Base directory for datasets
    transform: list of transforms or None, default None
        Transform the inputs.
    target_transform: list of transforms or None, default None
        Transform the outputs.
    train: bool, default True
        Load training data.
    augmentation: bool, default True
        Apply default data augmentation for training.

    Returns
    -------
    _dataset: Dataset
    """
    #dataset_dir = os.path.join(datasets_dir, dataset)
    dataset_dir = datasets_dir
    if transform is not None:
        raw_transforms = transform
    else:
        raw_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(MEANS[dataset], STDS[dataset])]

    if augmentation:
        if not train:
            print("Warning: using augmentation on eval data")
        raw_transforms = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip()
        ] + raw_transforms
    _transform = transforms.Compose(raw_transforms)

    if train:
        _dataset = _create_train_dataset(dataset, dataset_dir,
                                         transform=_transform,
                                         target_transform=target_transform)
    else:
        _dataset = _create_test_dataset(dataset, dataset_dir,
                                        transform=_transform,
                                        target_transform=target_transform)
    print(_dataset)
    return _dataset
