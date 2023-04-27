import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics as tm


class Classifier(pl.LightningModule):
    def __init__(self, architecture, num_classes, lr=0.01):
        super().__init__()

        self.save_hyperparameters()

        self.classifier = architecture
        self.num_classes = num_classes

        self.lr = lr

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
        return optim.Adam(self.parameters(), lr=self.lr)
