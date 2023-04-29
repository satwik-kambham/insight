import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from torchinfo import summary

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

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


params = {
    "data_dir": "datasets",
    "dataset": "CIFAR10",
    "architecture": "LeNet5",
    "val_batch_size": 64,
    "num_workers": 1,
    "accelerator": "auto",
}


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        session.report({"val_acc": trainer.callback_metrics["val_acc"].item()})


def objective(config):
    datamodule = DataModule(
        root=params["data_dir"],
        dataset=params["dataset"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=params["val_batch_size"],
        num_workers=params["num_workers"],
    )

    model = model_dict[params["architecture"]][0](
        *model_dict[params["architecture"]][1:], datamodule.num_classes
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
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        factor=config["factor"],
        patience=config["patience"],
        threshold=config["threshold"],
    )

    wandb_logger = WandbLogger(project="classifiers")
    wandb_logger.log_hyperparams(
        {
            "dataset": params["dataset"],
            "architecture": params["architecture"],
        }
    )
    tensorboard_logger = TensorBoardLogger("tensorboard_logs/")

    trainer = pl.Trainer(
        accelerator=params["accelerator"],
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[TuneReportCallback()],
    )

    trainer.fit(classifier, datamodule)


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

    search_space = {}

    algo = OptunaSearch()

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            search_alg=algo,
        ),
        run_config=air.RunConfig(
            log_to_file=True,
            stop={"training_iteration": params["epochs"]},
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    all_results = results.get_dataframe()
    all_results.to_csv("tune_results.csv")
    best_result = results.get_best_result()
    print(best_result.config)


if __name__ == "__main__":
    hyperopt()
