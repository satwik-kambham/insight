import torch
from torchvision import datasets, transforms


def load_data(root, hyperparameters, dataset="CIFAR10"):
    # Loading the dataset
    if dataset == "CIFAR10":
        train_dataset, val_dataset = load_CIFAR10(root)
    elif dataset == "Caltech256":
        train_dataset, val_dataset = load_Caltech256(root)
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

    return train_loader, val_loader


def load_CIFAR10(root):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
            transforms.CenterCrop(32),
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

    return train_dataset, val_dataset


def load_Caltech256(root):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
            transforms.CenterCrop(256),
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

    return train_dataset, val_dataset
