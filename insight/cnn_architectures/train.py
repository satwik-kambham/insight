import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics as tm

from dataloaders import load_data
from LeNet import LeNet5
from AlexNet import AlexNet
from VGG import VGG, VGG_A, VGG_B, VGG_C, VGG_D, VGG_E
from ResNet import ResNet, RESNET_18, RESNET_34, RESNET_50, RESNET_101, RESNET_152
from GoogLeNet import GoogLeNet

# Hyperparameters for training
hyperparameters = {
    "train_batch_size": 64,
    "val_batch_size": 64,
    "train_workers": 0,
    "val_workers": 0,
    "epochs": 10,
    "lr": 0.01,
    "img_shape": None,
    "architecture": "LeNet5",
    "dataset": "CIFAR10",
}


class Classifier(pl.LightningModule):
    def __init__(self, architecture, hyperparameters, num_classes):
        super().__init__()

        self.save_hyperparameters("hyperparameters", "num_classes")

        self.classifier = architecture
        self.hyperparameters = hyperparameters
        self.num_classes = num_classes

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
        return optim.Adam(self.parameters(), lr=self.hyperparameters["lr"])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data", type=str, default="./data", help="path to dataset"
    )

    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.01, help="learning rate"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="dataset to use - CIFAR10, Caltech256, Caltech101, fashion_MNIST, ...",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        default="LeNet5",
        help="CNN architecture to use - LeNet5, AlexNet, \
            VGG(11, 13, 16-1, 16, 19), GoogLeNet, \
                ResNet(18, 34, 50, 101, 152)",
    )

    # Image x and y dimensions
    parser.add_argument("--img_x", type=int, default=-1, help="image x dimension")
    parser.add_argument("--img_y", type=int, default=-1, help="image y dimension")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use for training / testing",
    )

    parser.add_argument(
        "--nightly", action="store_true", help="Use pytorch 2.0 to compile model"
    )

    parser.add_argument(
        "-s", "--save", action="store_true", help="Save model checkpoint"
    )

    args = parser.parse_args()
    return args


def get_model(architecture, num_classes):
    # Creating the model
    if architecture == "LeNet5":
        model = LeNet5(num_classes=num_classes)
    elif architecture == "AlexNet":
        model = AlexNet(num_classes=num_classes)
    elif architecture == "VGG11":
        model = VGG(VGG_A, num_classes=num_classes)
    elif architecture == "VGG13":
        model = VGG(VGG_B, num_classes=num_classes)
    elif architecture == "VGG16-1":
        model = VGG(VGG_C, num_classes=num_classes)
    elif architecture == "VGG16":
        model = VGG(VGG_D, num_classes=num_classes)
    elif architecture == "VGG19":
        model = VGG(VGG_E, num_classes=num_classes)
    elif architecture == "GoogLeNet":
        model = GoogLeNet(num_classes=num_classes)
    elif architecture == "ResNet18":
        model = ResNet(RESNET_18, num_classes=num_classes)
    elif architecture == "ResNet34":
        model = ResNet(RESNET_34, num_classes=num_classes)
    elif architecture == "ResNet50":
        model = ResNet(RESNET_50, block="bottleneck", num_classes=num_classes)
    elif architecture == "ResNet101":
        model = ResNet(RESNET_101, block="bottleneck", num_classes=num_classes)
    elif architecture == "ResNet152":
        model = ResNet(RESNET_152, block="bottleneck", num_classes=num_classes)
    else:
        raise NotImplementedError

    return model


def main():
    args = parse_args()

    hyperparameters["epochs"] = args.epochs
    hyperparameters["train_batch_size"] = args.batch_size
    hyperparameters["val_batch_size"] = args.batch_size
    hyperparameters["lr"] = args.learning_rate
    hyperparameters["architecture"] = args.architecture
    hyperparameters["dataset"] = args.dataset

    if args.img_x != -1 or args.img_y != -1:
        hyperparameters["img_shape"] = (args.img_x, args.img_y)

    # Loading the dataset
    train_loader, val_loader, num_classes, img_shape, num_channels = load_data(
        args.data,
        hyperparameters,
    )

    hyperparameters["img_shape"] = img_shape

    model = get_model(hyperparameters["architecture"], num_classes)

    test_input_size = (1, num_channels, *hyperparameters["img_shape"])
    test_input = torch.randn(test_input_size)
    _ = model(test_input)

    if hasattr(model, "_init_weights"):
        model._init_weights()

    if args.nightly:
        model = torch.compile(model)

    classifier = Classifier(model, hyperparameters, num_classes)

    if args.device == "auto":
        accelerator = "auto"
    elif args.device == "cpu":
        accelerator = None
    elif args.device == "gpu" or args.device == "cuda":
        accelerator = "gpu"
    else:
        raise NotImplementedError

    wandb_logger = WandbLogger(project="cnn-architectures")
    tensorboard_logger = TensorBoardLogger("tensorboard_logs/")

    trainer = pl.Trainer(
        accelerator=accelerator,
        enable_checkpointing=args.save,
        max_epochs=hyperparameters["epochs"],
        logger=[wandb_logger, tensorboard_logger],
    )

    trainer.fit(classifier, train_loader, val_loader)


if __name__ == "__main__":
    main()
