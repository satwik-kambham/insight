import torch
from torchvision import datasets, transforms


def load_data(root, hyperparameters):
    """Load the specified dataset and create the corresponding dataloaders
    Available datasets:
        - CIFAR10
        - Caltech256
        - Caltech101
        - fashion_MNIST

    Parameters
    ----------
    root : str
        Path where the dataset will be stored
    hyperparameters : dict
        dict of hyperparameters

    Returns
    -------
    tuple(DataLoader, DataLoader, int, tuple(int, int)))
        train and validation dataloaders, number of classes and image shape

    Raises
    ------
    ValueError
        If the specified dataset is not implemented
    """
    num_classes = 10
    dataset = hyperparameters["dataset"]
    img_shape = hyperparameters["img_shape"]

    # Loading the dataset
    if dataset == "CIFAR10":
        train_dataset, val_dataset, img_shape = load_CIFAR10(
            root, hyperparameters["img_shape"]
        )
        num_classes = 10
    elif dataset == "Caltech256":
        train_dataset, val_dataset, img_shape = load_Caltech256(
            root, hyperparameters["img_shape"]
        )
        num_classes = 257
    elif dataset == "Caltech101":
        train_dataset, val_dataset, img_shape = load_Caltech101(
            root, hyperparameters["img_shape"]
        )
        num_classes = 101
    elif dataset == "fashion_MNIST":
        train_dataset, val_dataset, img_shape = load_fashion_MNIST(
            root, hyperparameters["img_shape"]
        )
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

    # Creating the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters["train_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=hyperparameters["train_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hyperparameters["val_batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=hyperparameters["val_workers"],
    )

    return train_loader, val_loader, num_classes, img_shape


def load_CIFAR10(root, img_shape=(244, 244)):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
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

    return train_dataset, val_dataset, img_shape


def load_Caltech256(root, img_shape=(256, 256)):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
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

    return train_dataset, val_dataset, img_shape


def load_Caltech101(root, img_shape=(300, 200)):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
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

    return train_dataset, val_dataset, img_shape


def load_fashion_MNIST(root, img_shape=(244, 244)):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
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

    return train_dataset, val_dataset, img_shape
