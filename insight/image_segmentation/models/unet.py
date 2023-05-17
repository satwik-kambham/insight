import torch
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl
import torchmetrics as tm


class UNet(nn.Module):
    def __init__(self, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder_blocks = nn.ModuleList(
            [
                self.block(64),
                self.contracting_block(128),
                self.contracting_block(256),
                self.contracting_block(512),
            ]
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.expanding_block(1024),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList(
            [
                self.expanding_block(512),
                self.expanding_block(256),
                self.expanding_block(128),
                self.block(64),
            ]
        )

        # # Output layer
        self.output = nn.LazyConv2d(out_channels, kernel_size=1)

    def block(self, out_channels):
        return nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=3),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels, kernel_size=3),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
        )

    def contracting_block(self, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.block(out_channels),
        )

    def expanding_block(self, out_channels):
        return nn.Sequential(
            self.block(out_channels),
            nn.LazyConvTranspose2d(out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        encoder_features = []

        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # # Decoder
        for decoder_block, encoder_feature in zip(
            self.decoder_blocks, reversed(encoder_features)
        ):
            crop = (encoder_feature.size()[2] - x.size()[2]) // 2
            encoder_feature = encoder_feature[
                :, :, crop : crop + x.size()[2], crop : crop + x.size()[3]
            ]

            x = torch.cat([x, encoder_feature], dim=1)
            x = decoder_block(x)

        # Output
        output = self.output(x)
        return output


class UNetModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=0.01,
        momentum=0.99,
        weight_decay=0.0005,
        num_classes=21,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="model")

        self.model = model

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        self.iou = tm.JaccardIndex(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        pred = output.argmax(dim=1, keepdim=False)
        self.log("val_loss", loss)
        self.iou(pred.float(), target)
        self.log("val_iou", self.iou, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
        )
        return [optimizer], [
            {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        ]
