import torch
import numpy as np
from torchvision import datasets, transforms


def load_VOCSegmentation(data_dir):
    img_transforms = transforms.Compose(
        [
            transforms.Resize((572, 572)),
            transforms.ToTensor(),
        ]
    )

    def mask_to_tensor(mask):
        mask = np.array(mask)
        mask = mask == 0
        tensor_mask = torch.from_numpy(mask)
        tensor_mask = (
            torch.nn.functional.one_hot(tensor_mask, num_classes=2)
            .permute(2, 0, 1)
            .float()
        )
        return tensor_mask

    mask_transforms = transforms.Compose(
        [
            transforms.Resize((388, 388)),
            transforms.Lambda(lambda x: mask_to_tensor(x)),
        ]
    )

    train_dataset = datasets.VOCSegmentation(
        root=data_dir,
        image_set="train",
        transform=img_transforms,
        target_transform=mask_transforms,
        download=True,
    )

    val_dataset = datasets.VOCSegmentation(
        root=data_dir,
        image_set="val",
        transform=img_transforms,
        target_transform=mask_transforms,
        download=True,
    )

    return train_dataset, val_dataset
