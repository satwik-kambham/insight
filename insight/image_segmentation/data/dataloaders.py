import numpy as np

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

import albumentations as A

from ..utils.class_distribution import get_class_weights


class AugmentedDataset:
    def __init__(
        self,
        dataset,
        augmentation=None,
        img_transforms=None,
        mask_transforms=None,
    ):
        self.dataset = dataset
        self.augmentation = augmentation
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        img = np.array(img)
        mask = np.array(mask)

        if self.augmentation:
            augmented = self.augmentation(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if self.img_transforms:
            img = self.img_transforms(img)

        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return img, mask


def get_augmentation(img_shape):
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(*img_shape),
            A.Rotate(),
            A.HorizontalFlip(),
            A.RGBShift(),
            A.Blur(),
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.Resize(*img_shape),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(*img_shape),
        ]
    )

    return train_transform, val_transform


def load_OxfordIIITPetDataset(data_dir, img_shape=(128, 128)):
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def mask_to_tensor(mask):
        mask = mask - 1
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_mask

    mask_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: mask_to_tensor(x)),
        ]
    )

    dataset = datasets.OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="segmentation",
        download=True,
    )

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_augmentation, val_augmentation = get_augmentation(img_shape)
    train_dataset = AugmentedDataset(
        train_dataset, train_augmentation, img_transforms, mask_transforms
    )
    val_dataset = AugmentedDataset(
        val_dataset, val_augmentation, img_transforms, mask_transforms
    )

    num_classes = 3
    class_weights = get_class_weights(train_dataset, val_dataset, num_classes)

    return train_dataset, val_dataset, num_classes, class_weights


def load_VOCSegmentationDataset(data_dir, simple=False, img_shape=(128, 128)):
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def mask_to_tensor(mask):
        mask = np.array(mask)
        mask[mask == 255] = 21
        if simple:
            mask[mask == 21] = 0
            mask[mask != 0] = 1
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_mask

    mask_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: mask_to_tensor(x)),
        ]
    )

    dataset = datasets.VOCSegmentation(
        root=data_dir,
        image_set="trainval",
        download=True,
    )

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_augmentation, val_augmentation = get_augmentation(img_shape)
    train_dataset = AugmentedDataset(
        train_dataset, train_augmentation, img_transforms, mask_transforms
    )
    val_dataset = AugmentedDataset(
        val_dataset, val_augmentation, img_transforms, mask_transforms
    )

    num_classes = 22
    if simple:
        num_classes = 2
    class_weights = get_class_weights(train_dataset, val_dataset, num_classes)

    if num_classes == 22:
        class_weights[0] = 0.1

    return train_dataset, val_dataset, num_classes, class_weights
