from torchvision import datasets, transforms


def load_VOCSegmentation(data_dir):
    img_transforms = transforms.Compose([])
    mask_transforms = transforms.Compose([])

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
