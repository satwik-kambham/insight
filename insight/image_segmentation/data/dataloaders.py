from torchvision import datasets, transforms


def load_VOCSegmentation(data_dir):
    img_transforms = transforms.Compose(
        [
            transforms.Resize((572, 572)),
            transforms.ToTensor(),
        ]
    )
    mask_transforms = transforms.Compose(
        [
            transforms.Resize((388, 388)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.VOCSegmentation(
        root=data_dir,
        image_set="train",
        transform=img_transforms,
        target_transform=mask_transforms,
    )

    val_dataset = datasets.VOCSegmentation(
        root=data_dir,
        image_set="val",
        transform=img_transforms,
        target_transform=mask_transforms,
    )

    return train_dataset, val_dataset
