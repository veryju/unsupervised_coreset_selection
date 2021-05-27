import os
import sys
import numpy as np
from PIL import Image

import torchvision
from typing import Optional, List, Callable, Any
from torch.utils.data import Dataset, Subset, DataLoader

from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg, download_url

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

Transform = Callable[[Any], Any]

DATASETS = [ 'tiny' ]

MEANS = { 'tiny': (0.485, 0.456, 0.406) }

STDS = { 'tiny': (0.229, 0.224, 0.225) }

#################################################################################          
################################### TinyImageNet ################################
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class ImageNetDatasetFolder(VisionDataset):


    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(ImageNetDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.targets = np.array(self.targets)
        self.targets = self.targets.astype(np.int)
        
        total_core_set = 0
        print(self.targets)
        for i in range(200):
            print(i, ' class : ', 'data count :', (self.targets==i).sum())
            total_core_set = total_core_set+ (self.targets==i).sum()
        print('the size of total core set  : ', total_core_set)   

    def _find_classes(self, dir):

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class TinyImageNet(ImageNetDatasetFolder):

    def __init__(self, root, split='train', transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        self.split = split
        root = os.path.join(root,self.split)
        
        super(TinyImageNet, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples 


def _create_test_dataset(dataset, dataset_dir, transform,
                         target_transform=None):
    if dataset == 'tiny':
        test_dataset = TinyImageNet(root=dataset_dir, split='val',
                                    transform=transform,
                                    target_transform=target_transform)
    else:
        raise NotImplementedError(f'{dataset} is not an available option.')
    return test_dataset


def _create_train_dataset(dataset, dataset_dir, transform,
                          target_transform=None):
    if dataset == 'tiny':
        train_dataset = TinyImageNet(root=dataset_dir, split='train',
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
    dataset_dir = os.path.join(datasets_dir, dataset)
    #dataset_dir = datasets_dir
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
            transforms.RandomCrop(64, padding=4),
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
