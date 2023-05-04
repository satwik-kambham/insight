import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary
import lightning.pytorch as pl
import torchmetrics as tm

from .LeNet import LeNet5
from .AlexNet import AlexNet
from .VGG import VGG_A, VGG_B, VGG_C, VGG_D, VGG_E, VGG
from .GoogLeNet import GoogLeNet
from .ResNet import (
    RESNET_18,
    RESNET_34,
    RESNET_50,
    RESNET_101,
    RESNET_152,
    ResNet,
)


model_dict = {
    "LeNet5": (LeNet5,),
    "AlexNet": (AlexNet,),
    "VGG_A": (VGG, VGG_A),
    "VGG_B": (VGG, VGG_B),
    "VGG_C": (VGG, VGG_C),
    "VGG_D": (VGG, VGG_D),
    "VGG_E": (VGG, VGG_E),
    "GoogLeNet": (GoogLeNet,),
    "ResNet18": (ResNet, RESNET_18, "simple"),
    "ResNet34": (ResNet, RESNET_34, "simple"),
    "ResNet50": (ResNet, RESNET_50, "bottleneck"),
    "ResNet101": (ResNet, RESNET_101, "bottleneck"),
    "ResNet152": (ResNet, RESNET_152, "bottleneck"),
}


class Classifier(pl.LightningModule):
    def __init__(
        self,
        architecture,
        num_classes,
        num_channels,
        img_shape,
        labels,
        lr=0.01,
        weight_decay=0.0,
        factor=0.1,
        patience=10,
        threshold=0.0001,
        compile=False,
    ):
        super().__init__()

        self.save_hyperparameters()

        model = model_dict[architecture][0](
            *model_dict[architecture][1:],
            num_classes,
        )

        test_input_size = (1, num_channels, *img_shape)
        test_input = torch.randn(test_input_size)
        _ = model(test_input)

        if hasattr(model, "_init_weights"):
            model._init_weights()

        summary(model, input_data=test_input)

        if compile:
            model = torch.compile(model)

        self.classifier = model
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.img_shape = img_shape
        self.labels = labels

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.threshold = threshold

        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = tm.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.classifier(x)

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
        self.accuracy(pred, target)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
        )
        # lr_scheduler2 = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     epochs=self.trainer.max_epochs,
        #     steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
        # )
        return [optimizer], [
            {
                "scheduler": lr_scheduler1,
                "monitor": "val_loss",
                "interval": "epoch",
            },
            # {
            #     "scheduler": lr_scheduler2,
            #     "interval": "step",
            # },
        ]
