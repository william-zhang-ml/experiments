import os
from typing import Tuple
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms


my_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(my_dir, 'data')


def get_cifar100() -> Tuple[CIFAR100, CIFAR100]:
    """ Get dataset objects for CIFAR100.

    :return: CIFAR100 train and CIFAR100 validation
    :rtype:  Tuple[CIFAR100, CIFAR100]
    """
    try:
        train = CIFAR100(
            root=data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False)
    except RuntimeError:
        train = CIFAR100(
            root=data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=True)  # download if needed

    val = CIFAR100(
        root=data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=False)

    return train, val
