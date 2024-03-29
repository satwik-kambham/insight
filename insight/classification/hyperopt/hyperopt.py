import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger

import optuna
from .optuna_integration import PyTorchLightningPruningCallback

import click

from ..data.datamodule import DataModule
from ..data.dataloaders import load_data
from ..models.classifier import Classifier


params = {
    "data_dir": "datasets",
    "dataset": "CIFAR10",
    "architecture": "LeNet5",
    "val_batch_size": 64,
    "num_workers": 1,
    "accelerator": "auto",
}


def objective(trial: optuna.trial.Trial):
    train_batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    factor = trial.suggest_float("factor", 0.1, 0.9)
    patience = trial.suggest_int("patience", 1, 10)
    threshold = trial.suggest_float("threshold", 1e-5, 1e-1, log=True)

    datamodule = DataModule(
        root=params["data_dir"],
        dataset=params["dataset"],
        train_batch_size=train_batch_size,
        val_batch_size=params["val_batch_size"],
        num_workers=params["num_workers"],
    )

    classifier = Classifier(
        architecture=params["architecture"],
        num_classes=datamodule.num_classes,
        num_channels=datamodule.num_channels,
        img_shape=datamodule.img_shape,
        labels=datamodule.labels,
        lr=lr,
        weight_decay=weight_decay,
        factor=factor,
        patience=patience,
        threshold=threshold,
        compile=compile,
    )

    # wandb_logger = WandbLogger(project="classifiers")
    # wandb_logger.log_hyperparams(params)
    # tensorboard_logger = TensorBoardLogger("tensorboard_logs/")

    trainer = pl.Trainer(
        accelerator=params["accelerator"],
        max_epochs=params["epochs"],
        # logger=[wandb_logger, tensorboard_logger],
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        enable_progress_bar=True,
    )

    trainer.fit(classifier, datamodule)

    return trainer.callback_metrics["val_loss"].item()


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
@click.option("--epochs", type=int, default=10, help="Training batch size")
@click.option("--val_batch_size", type=int, default=64, help="Validation batch size")
@click.option("--num_workers", type=int, default=4, help="Number of workers")
@click.option(
    "--accelerator", type=str, default="auto", help="Accelerator: auto, cpu, gpu, tpu"
)
def hyperopt(
    data_dir,
    dataset,
    architecture,
    epochs,
    val_batch_size,
    num_workers,
    accelerator,
):
    params["data_dir"] = data_dir
    params["dataset"] = dataset
    params["architecture"] = architecture
    params["epochs"] = epochs
    params["val_batch_size"] = val_batch_size
    params["num_workers"] = num_workers
    params["accelerator"] = accelerator

    # Download dataset to data_dir
    load_data(params["data_dir"], params["dataset"], None)

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective)

    best_trial = study.best_trial

    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    all_trails = study.trials_dataframe()
    all_trails.to_csv("trails.csv")


if __name__ == "__main__":
    hyperopt()
