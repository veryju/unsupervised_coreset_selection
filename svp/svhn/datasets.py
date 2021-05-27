import os
import numpy as np

from typing import Optional, List, Callable, Any
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

Transform = Callable[[Any], Any]

DATASETS = [ 'svhn' ]

MEANS = { 'svhn': (0.5, 0.5, 0.5) }

STDS = { 'svhn': (0.5, 0.5, 0.5) }


def _create_test_dataset(dataset, dataset_dir, transform,
                         target_transform=None):
    if dataset == 'svhn':
        test_dataset = datasets.SVHN(root=dataset_dir, split='test',
                                        download=True,
                                        transform=transform,
                                        target_transform=target_transform)
    else:
        raise NotImplementedError(f'{dataset} is not an available option.')
    return test_dataset


def _create_train_dataset(dataset, dataset_dir, transform,
                          target_transform=None):
    if dataset == 'svhn':
        train_dataset = datasets.SVHN(root=dataset_dir, split='train',
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
            transforms.RandomCrop(32, padding=4),
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
