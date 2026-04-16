from typing import Tuple

from torchvision import datasets, transforms
from torchvision.datasets import FakeData


def _fallback_fakedata(channels: int, image_size: int, num_classes: int):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_set = FakeData(
        size=6000,
        image_size=(channels, image_size, image_size),
        num_classes=num_classes,
        transform=tfm,
        random_offset=0,
    )
    test_set = FakeData(
        size=1000,
        image_size=(channels, image_size, image_size),
        num_classes=num_classes,
        transform=tfm,
        random_offset=10000,
    )
    # FakeData does not provide targets field by default, add one for partitioning.
    train_set.targets = [train_set[i][1] for i in range(len(train_set))]
    test_set.targets = [test_set[i][1] for i in range(len(test_set))]
    return train_set, test_set


def build_dataset(name: str, root: str, offline_fallback: bool = False) -> Tuple[object, object, int, int]:
    if name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        try:
            if offline_fallback:
                raise RuntimeError("offline_fallback=true")
            train_set = datasets.MNIST(root=root, train=True, transform=tfm, download=True)
            test_set = datasets.MNIST(root=root, train=False, transform=tfm, download=True)
        except Exception:
            if offline_fallback:
                train_set, test_set = _fallback_fakedata(channels=1, image_size=28, num_classes=10)
            else:
                raise
        return train_set, test_set, 1, 10

    if name == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor()])
        try:
            if offline_fallback:
                raise RuntimeError("offline_fallback=true")
            train_set = datasets.CIFAR10(root=root, train=True, transform=tfm, download=True)
            test_set = datasets.CIFAR10(root=root, train=False, transform=tfm, download=True)
        except Exception:
            if offline_fallback:
                train_set, test_set = _fallback_fakedata(channels=3, image_size=32, num_classes=10)
            else:
                raise
        return train_set, test_set, 3, 10

    if name == "cifar100":
        tfm = transforms.Compose([transforms.ToTensor()])
        try:
            if offline_fallback:
                raise RuntimeError("offline_fallback=true")
            train_set = datasets.CIFAR100(root=root, train=True, transform=tfm, download=True)
            test_set = datasets.CIFAR100(root=root, train=False, transform=tfm, download=True)
        except Exception:
            if offline_fallback:
                train_set, test_set = _fallback_fakedata(channels=3, image_size=32, num_classes=100)
            else:
                raise
        return train_set, test_set, 3, 100

    raise ValueError(f"Unsupported dataset: {name}")
