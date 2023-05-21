import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from torchinfo import summary

import click

from ..data.datamodule import VOCSegmentationDataModule
from ..models.unet import UNet, UNetModule


@click.command()
@click.option("--data_dir", type=str, default="datasets", help="Path to data directory")
@click.option("--batch_size", type=int, default=1, help="Batch size")
@click.option("--num_workers", type=int, default=4, help="Number of workers")
@click.option("--lr", type=float, default=0.01, help="Learning rate")
@click.option("--momentum", type=float, default=0.99, help="Momentum")
@click.option("--weight_decay", type=float, default=0.0005, help="Weight decay")
@click.option("--epochs", type=int, default=20, help="Number of epochs")
@click.option(
    "--accelerator", type=str, default="auto", help="Accelerator: auto, cpu, gpu, tpu"
)
@click.option("--compile", type=bool, default=False, help="Compile model")
def train(
    data_dir,
    batch_size,
    num_workers,
    lr,
    momentum,
    weight_decay,
    epochs,
    accelerator,
    compile,
):
    pl.seed_everything(42, workers=True)

    datamodule = VOCSegmentationDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = UNet(2)

    test_input_shape = (1, 3, 572, 572)
    test_input = torch.randn(test_input_shape)
    _ = model(test_input)

    summary(model, input_size=test_input_shape)

    if compile:
        model = model.compile()

    unet_module = UNetModule(
        model=model,
        num_classes=2,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    wandb_logger = WandbLogger(project="semantic_segmentation")
    tensorboard_logger = TensorBoardLogger("tensorboard_logs/")

    lr_monitor = LearningRateMonitor(log_momentum=True)

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[lr_monitor],
    )

    trainer.fit(unet_module, datamodule=datamodule)


if __name__ == "__main__":
    train()
