import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from torchinfo import summary

import click

from ..data.datamodule import DataModule
from ..models.classifier import Classifier

from ..models.LeNet import LeNet5
from ..models.AlexNet import AlexNet
from ..models.VGG import VGG_A, VGG_B, VGG_C, VGG_D, VGG_E, VGG
from ..models.GoogLeNet import GoogLeNet
from ..models.ResNet import (
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


@click.command()
@click.option("--data_dir", type=str, default="datasets", help="Path to data directory")
@click.option(
    "--dataset",
    type=str,
    default="CIFAR10",
    help="""Available datasets:
    - CIFAR10
    - Caltech256
    - Caltech101
    - fashion_MNIST
    """,
)
@click.option(
    "--architecture",
    type=str,
    default="LeNet5",
    help="""Available models:
    - LeNet5
    - AlexNet
    - VGG-(11, 13, 16, 19)
    - GoogLeNet
    - ResNet-(18, 34, 50, 101, 152)
    """,
)
@click.option("--train_batch_size", type=int, default=64, help="Training batch size")
@click.option("--val_batch_size", type=int, default=64, help="Validation batch size")
@click.option("--num_workers", type=int, default=4, help="Number of workers")
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option("--weight_decay", type=float, default=0.0, help="Weight decay")
@click.option("--factor", type=float, default=0.1, help="Factor")
@click.option("--patience", type=int, default=3, help="Patience")
@click.option("--threshold", type=float, default=0.0001, help="Threshold")
@click.option("--epochs", type=int, default=20, help="Number of epochs")
@click.option(
    "--accelerator", type=str, default="auto", help="Accelerator: auto, cpu, gpu, tpu"
)
@click.option("--compile", type=bool, default=False, help="Compile model")
def train(
    data_dir,
    dataset,
    architecture,
    train_batch_size,
    val_batch_size,
    num_workers,
    lr,
    weight_decay,
    factor,
    patience,
    threshold,
    epochs,
    accelerator,
    compile,
):
    datamodule = DataModule(
        root=data_dir,
        dataset=dataset,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
    )

    model = model_dict[architecture][0](
        *model_dict[architecture][1:], datamodule.num_classes
    )

    test_input_size = (1, datamodule.num_channels, *datamodule.img_shape)
    test_input = torch.randn(test_input_size)
    _ = model(test_input)

    if hasattr(model, "_init_weights"):
        model._init_weights()

    summary(model, input_data=test_input)

    if compile:
        model = torch.compile(model)

    classifier = Classifier(
        model,
        datamodule.num_classes,
        lr=lr,
        weight_decay=weight_decay,
        factor=factor,
        patience=patience,
        threshold=threshold,
    )

    wandb_logger = WandbLogger(project="classifiers")
    wandb_logger.log_hyperparams(
        {
            "dataset": dataset,
            "architecture": architecture,
        }
    )
    tensorboard_logger = TensorBoardLogger("tensorboard_logs/")

    lr_monitor = LearningRateMonitor(log_momentum=True)

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[lr_monitor],
    )

    trainer.fit(classifier, datamodule=datamodule)


if __name__ == "__main__":
    train()
