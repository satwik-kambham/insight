import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import torchmetrics as tm
from torchinfo import summary

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
        x = nn.functional.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = self.down2(x)
        x = nn.functional.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = self.down3(x)
        x = nn.functional.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = self.down4(x)

        # Upsampling path
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        x = self.output(x)

        return x


class UNetResNetBackbone(nn.Module):
    def __init__(self, out_channels, pretrained=None):
        super(UNet, self).__init__()
        self.pretrained = pretrained

        if pretrained:
            if pretrained == "resnet18":
                pretrained_model = torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.DEFAULT
                )
            elif pretrained == "resnet34":
                pretrained_model = torchvision.models.resnet34(
                    weights=torchvision.models.ResNet34_Weights.DEFAULT
                )
            elif pretrained == "resnet50":
                pretrained_model = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.DEFAULT
                )
            elif pretrained == "resnet101":
                pretrained_model = torchvision.models.resnet101(
                    weights=torchvision.models.ResNet101_Weights.DEFAULT
                )
            elif pretrained == "resnet152":
                pretrained_model = torchvision.models.resnet152(
                    weights=torchvision.models.ResNet152_Weights.DEFAULT
                )
            else:
                raise ValueError(
                    """Invalid pretrained model.
                    Must be one of:
                    resnet18, resnet34, resnet50, resnet101, resnet152"""
                )

        self.down1 = DoubleConv(64)
        self.down2 = pretrained_model.layer2
        self.down3 = pretrained_model.layer3
        self.down4 = pretrained_model.layer4

        self.up1 = nn.LazyConvTranspose2d(512, kernel_size=2, stride=2)
        self.up2 = nn.LazyConvTranspose2d(256, kernel_size=2, stride=2)
        self.up3 = nn.LazyConvTranspose2d(128, kernel_size=2, stride=2)
        self.up4 = DoubleConv(64)

        self.output = nn.LazyConv2d(out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Upsampling path
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
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
        pretrained=None,
        class_weights=None,
        lr=0.01,
        weight_decay=0.0001,
        compile=False,
    ):
        super().__init__()

        self.save_hyperparameters()

        if num_classes == 2:
            num_classes = 1

        if pretrained:
            self.model = UNetResNetBackbone(num_classes, pretrained=pretrained)
        else:
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

        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        if num_classes == 1:
            self.accuracy = tm.Accuracy(task="binary")
            self.iou = tm.JaccardIndex(task="binary")
        else:
            self.accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)
            self.iou = tm.JaccardIndex(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        if self.num_classes == 1:
            output = output.squeeze(1)
            target = target.float()
        loss = self.criterion(output, target)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        if self.num_classes == 1:
            output = output.squeeze(1)
            target = target.float()
        loss = self.criterion(output, target)
        self.log("val_loss", loss)

        if self.num_classes == 1:
            pred = output
            target = target.long()
        else:
            pred = output.argmax(dim=1, keepdim=False)

        self.val_pred = output
        self.val_target = target

        self.accuracy(pred, target)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True)

        self.iou(pred, target)
        self.log("val_iou", self.iou, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        if self.num_classes == 1:
            self.val_pred = torch.sigmoid(self.val_pred.unsqueeze(1))
            mask_image = generate_mask(self.val_pred, self.num_classes + 1, False)
        else:
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
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
