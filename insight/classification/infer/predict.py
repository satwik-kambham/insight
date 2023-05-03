import click

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
@click.option("--ckpt_file", type=str, help="Path to checkpoint file")
def infer(ckpt_file):
    classifier = Classifier.load_from_checkpoint(
        checkpoint_path=ckpt_file,
    )


if __name__ == "__main__":
    infer()
