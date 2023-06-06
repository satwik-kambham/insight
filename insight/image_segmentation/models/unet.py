import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import torchmetrics as tm
from torchinfo import summary

from .loss import dice_loss
from ..utils.mask import generate_mask


class DoubleConv(nn.Module):
    def __init__(self, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_channels):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(64)
        self.down2 = DoubleConv(128)
        self.down3 = DoubleConv(256)
        self.down4 = DoubleConv(512)
        self.up1 = nn.LazyConvTranspose2d(512, kernel_size=2, stride=2)
        self.up2 = nn.LazyConvTranspose2d(256, kernel_size=2, stride=2)
        self.up3 = nn.LazyConvTranspose2d(128, kernel_size=2, stride=2)
        self.up4 = DoubleConv(64)

        self.output = nn.LazyConv2d(out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        x1 = self.down1(x)
        x2 = nn.functional.max_pool2d(x1, kernel_size=2, stride=2)
        x3 = self.down2(x2)
        x4 = nn.functional.max_pool2d(x3, kernel_size=2, stride=2)
        x5 = self.down3(x4)
        x6 = nn.functional.max_pool2d(x5, kernel_size=2, stride=2)
        x7 = self.down4(x6)

        # Upsampling path
        x = self.up1(x7)
        x = torch.cat([x, x5], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        x = self.output(x)

        return x


class UNetModule(pl.LightningModule):
    def __init__(
        self,
        inp_size,
        num_classes=3,
        lr=0.01,
        weight_decay=0.0001,
        compile=False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = UNet(num_classes)

        test_input_shape = (1, 3, inp_size, inp_size)
        test_input = torch.randn(test_input_shape)
        _ = self.model(test_input)

        summary(self.model, input_size=test_input_shape)

        if compile:
            self.model = self.model.compile()

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.iou = tm.JaccardIndex(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        loss += dice_loss(output, target, self.num_classes)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        loss += dice_loss(output, target, self.num_classes)
        self.log("val_loss", loss)

        self.val_pred = output
        self.val_target = target

        pred = output.argmax(dim=1, keepdim=False)
        self.accuracy(pred, target)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True)

        self.iou(pred, target)
        self.log("val_iou", self.iou, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        mask_image = generate_mask(self.val_pred, self.num_classes + 1)
        target_mask = generate_mask(self.val_target, self.num_classes + 1, False)

        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                np_mask_image = np.array(mask_image.convert("RGB"))
                np_mask_image = np_mask_image.transpose(2, 0, 1)
                logger.experiment.add_image(
                    "val_mask", np_mask_image, global_step=self.current_epoch
                )

            if isinstance(logger, WandbLogger):
                logger.log_image(key="val_mask", images=[mask_image])
                logger.log_image(key="val_target", images=[target_mask])

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
