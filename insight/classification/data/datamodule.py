from torch.utils.data import DataLoader

import lightning.pytorch as pl

from .dataloaders import load_data


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        dataset,
        img_shape=None,
        train_batch_size=32,
        val_batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.root = root
        self.dataset = dataset
        self.img_shape = img_shape
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        (
            self.train_dataset,
            self.val_dataset,
            self.num_classes,
            self.img_shape,
            self.num_channels,
        ) = load_data(self.root, self.dataset, self.img_shape)

        self.save_hyperparameters(ignore=["img_shape"])

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )
