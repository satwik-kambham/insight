import torch
from torchvision import datasets, transforms


def load_data(root, dataset, img_shape):
    """Load the specified dataset and create the corresponding dataloaders

    Parameters
    ----------
    root : str
        Path where the dataset will be stored
    dataset : str
        Available datasets:
        - CIFAR10
        - Caltech256
        - Caltech101
        - fashion_MNIST
    img_shape : tuple(int, int)
        Image shape
        If None, the default shape for the dataset will be used

    Returns
    -------
    tuple(torch.utils.data.Dataset, torch.utils.data.Dataset, int, tuple(int, int), int)
        train and validation dataloaders, number of classes, image shape and
        number of channels

    Raises
    ------
    ValueError
        If the specified dataset is not implemented
    """
    num_classes = 10
    num_channels = 3

    # Loading the dataset
    if dataset == "CIFAR10":
        train_dataset, val_dataset, img_shape, labels = load_CIFAR10(root, img_shape)
        num_classes = 10
    elif dataset == "Caltech256":
        train_dataset, val_dataset, img_shape, labels = load_Caltech256(root, img_shape)
        num_classes = 257
    elif dataset == "Caltech101":
        train_dataset, val_dataset, img_shape, labels = load_Caltech101(root, img_shape)
        num_classes = 101
    elif dataset == "fashion_MNIST":
        train_dataset, val_dataset, img_shape, labels = load_fashion_MNIST(
            root, img_shape
        )
        num_classes = 10
        num_channels = 1
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    return train_dataset, val_dataset, num_classes, img_shape, num_channels, labels


img_augmentation = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
    ]
)


def load_CIFAR10(root, img_shape):
    if img_shape is None:
        img_shape = (32, 32)

    # Loading the dataset
    img_transform = transforms.Compose(
        [
            img_augmentation,
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    label_transform = transforms.Compose([])
    train_dataset = datasets.CIFAR10(
        root,
        train=True,
        transform=img_transform,
        target_transform=label_transform,
        download=True,
    )

    val_dataset = datasets.CIFAR10(
        root,
        train=False,
        transform=img_transform,
        target_transform=label_transform,
        download=True,
    )

    labels = train_dataset.classes

    return train_dataset, val_dataset, img_shape, labels


def load_Caltech256(root, img_shape):
    if img_shape is None:
        img_shape = (256, 256)

    # Loading the dataset
    img_transform = transforms.Compose(
        [
            img_augmentation,
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ]
    )
    label_transform = transforms.Compose([])
    dataset = datasets.Caltech256(
        root,
        transform=img_transform,
        target_transform=label_transform,
        download=True,
    )

    # Splitting the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

    labels = dataset.categories

    return train_dataset, val_dataset, img_shape, labels


def load_Caltech101(root, img_shape):
    if img_shape is None:
        img_shape = (300, 200)

    # Loading the dataset
    img_transform = transforms.Compose(
        [
            img_augmentation,
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ]
    )
    label_transform = transforms.Compose([])
    dataset = datasets.Caltech101(
        root,
        transform=img_transform,
        target_transform=label_transform,
        download=True,
    )

    # Splitting the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

    labels = dataset.categories

    return train_dataset, val_dataset, img_shape, labels


def load_fashion_MNIST(root, img_shape):
    if img_shape is None:
        img_shape = (28, 28)

    # Loading the dataset
    img_transform = transforms.Compose(
        [
            img_augmentation,
            transforms.Resize(img_shape),
            transforms.ToTensor(),
        ]
    )
    label_transform = transforms.Compose([])
    train_dataset = datasets.FashionMNIST(
        root,
        train=True,
        transform=img_transform,
        target_transform=label_transform,
        download=True,
    )

    val_dataset = datasets.FashionMNIST(
        root,
        train=False,
        transform=img_transform,
        target_transform=label_transform,
        download=True,
    )

    labels = train_dataset.classes

    return train_dataset, val_dataset, img_shape, labels
