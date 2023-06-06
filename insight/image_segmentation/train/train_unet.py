import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import click

from ..data.datamodule import SegmentationDataModule
from ..models.unet import UNetModule


def train(
    data_dir,
    dataset,
    inp_size,
    batch_size,
    num_workers,
    lr,
    weight_decay,
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
    elif dataset == "VOCSegmentationSimple":
        NUM_CLASSES = 2

    unet_module = UNetModule(
        num_classes=NUM_CLASSES,
        inp_size=inp_size,
        lr=lr,
        weight_decay=weight_decay,
        compile=compile,
    )

    wandb_logger = WandbLogger(project="semantic_segmentation")
    wandb_logger.log_hyperparams(
        {
            "dataset": dataset,
        }
    )
    tensorboard_logger = TensorBoardLogger("tensorboard_logs/")

    lr_monitor = LearningRateMonitor(log_momentum=True)

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(unet_module, datamodule=datamodule)


@click.command()
@click.option("--data_dir", type=str, default="datasets", help="Path to data directory")
@click.option(
    "--dataset",
    type=str,
    default="OxfordIIITPet",
    help="Dataset name: OxfordIIITPet, VOCSegmentation, VOCSegmentationSimple",
)
@click.option("--inp_size", type=int, default=128, help="Input size")
@click.option("--batch_size", type=int, default=1, help="Batch size")
@click.option("--num_workers", type=int, default=4, help="Number of workers")
@click.option("--lr", type=float, default=0.01, help="Learning rate")
@click.option("--weight_decay", type=float, default=0.0001, help="Weight decay")
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
