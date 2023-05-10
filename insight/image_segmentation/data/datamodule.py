from torch.utils.data import DataLoader
from torchvision import datasets

import lightning.pytorch as pl

from .dataloaders import load_VOCSegmentation


class VOCSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=16,
        num_workers=0,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.save_hyperparameters()

    def prepare_data(self):
        datasets.VOCSegmentation(
            root=self.data_dir,
            image_set="trainval",
            download=True,
        )

    def setup(self, stage):
        self.train_dataset, self.val_dataset = load_VOCSegmentation(self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )
