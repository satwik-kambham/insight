import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from torchinfo import summary

import click

from ..data.datamodule import SegmentationDataModule
from ..models.unet import UNet, UNetModule


def train(
    data_dir,
    dataset,
    inp_size,
    batch_size,
    num_workers,
    lr,
    epochs,
    accelerator,
    compile,
):
    pl.seed_everything(42, workers=True)

    datamodule = SegmentationDataModule(
        data_dir=data_dir,
        dataset=dataset,
        inp_size=inp_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if dataset == "OxfordIIITPet":
        NUM_CLASSES = 3
    elif dataset == "VOCSegmentation":
        NUM_CLASSES = 22

    model = UNet(NUM_CLASSES)

    test_input_shape = (1, 3, inp_size, inp_size)
    test_input = torch.randn(test_input_shape)
    _ = model(test_input)

    summary(model, input_size=test_input_shape)

    if compile:
        model = model.compile()

    unet_module = UNetModule(
        model=model,
        num_classes=2,
        lr=lr,
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


@click.command()
@click.option("--data_dir", type=str, default="datasets", help="Path to data directory")
@click.option(
    "--dataset",
    type=str,
    default="OxfordIIITPet",
    help="Dataset name: OxfordIIITPet, VOCSegmentation",
)
@click.option("--inp_size", type=int, default=128, help="Input size")
@click.option("--batch_size", type=int, default=1, help="Batch size")
@click.option("--num_workers", type=int, default=4, help="Number of workers")
@click.option("--lr", type=float, default=0.01, help="Learning rate")
@click.option("--epochs", type=int, default=20, help="Number of epochs")
@click.option(
    "--accelerator", type=str, default="auto", help="Accelerator: auto, cpu, gpu, tpu"
)
@click.option("--compile", type=bool, default=False, help="Compile model")
def train_cli(
    data_dir,
    dataset,
    inp_size,
    batch_size,
    num_workers,
    lr,
    epochs,
    accelerator,
    compile,
):
    train(
        data_dir,
        dataset,
        inp_size,
        batch_size,
        num_workers,
        lr,
        epochs,
        accelerator,
        compile,
    )


if __name__ == "__main__":
    train_cli()
