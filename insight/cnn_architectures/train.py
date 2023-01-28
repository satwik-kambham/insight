import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary
import wandb

from dataloaders import load_data
from LeNet import LeNet5
from AlexNet import AlexNet
from VGG import *

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


def train_loop(model, loss_fn, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx * data.shape[0]
    return train_loss


def val_loop(model, loss_fn, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= batch_idx * data.shape[0]
    val_accuracy = correct / (batch_idx * data.shape[0])
    return val_loss, val_accuracy


def train(model, loss_fn, optimizer, train_loader, val_loader, epochs, device):
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss = train_loop(model, loss_fn, optimizer, train_loader, device)
        val_loss, val_accuracy = val_loop(model, loss_fn, val_loader, device)
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )
    return train_loss, val_loss, val_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data", type=str, default="./data", help="path to dataset"
    )
    parser.add_argument("-s", type=bool, default=False, help="save model")

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
        help="CNN architecture to use - LeNet5, AlexNet, VGG(11, 13, 16-1, 16, 19)",
    )

    # Image x and y dimensions
    parser.add_argument("--img_x", type=int, default=-1, help="image x dimension")
    parser.add_argument("--img_y", type=int, default=-1, help="image y dimension")

    parser.add_argument(
        "--device", type=str, default="cpu", help="device to use for training / testing"
    )

    args = parser.parse_args()

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

    wandb.init(project="cnn-architectures", config=hyperparameters)

    device = torch.device(args.device)

    # Creating the model
    if hyperparameters["architecture"] == "LeNet5":
        model = LeNet5(num_classes=num_classes)
    elif hyperparameters["architecture"] == "AlexNet":
        model = AlexNet(num_classes=num_classes)
    elif hyperparameters["architecture"] == "VGG11":
        model = VGG(VGG_A, num_classes=num_classes)
    elif hyperparameters["architecture"] == "VGG13":
        model = VGG(VGG_B, num_classes=num_classes)
    elif hyperparameters["architecture"] == "VGG16-1":
        model = VGG(VGG_C, num_classes=num_classes)
    elif hyperparameters["architecture"] == "VGG16":
        model = VGG(VGG_D, num_classes=num_classes)
    elif hyperparameters["architecture"] == "VGG19":
        model = VGG(VGG_E, num_classes=num_classes)
    else:
        raise NotImplementedError
    test_input_size = (1, num_channels, *hyperparameters["img_shape"])
    test_input = torch.randn(test_input_size)
    test_output = model(test_input)

    model = torch.compile(model)

    # Printing the model summary
    model.eval()
    summary(model, input_size=test_input_size)

    # Defining the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"])

    wandb.watch(model)

    # Training the model
    train_loss, val_loss, val_accuracy = train(
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        hyperparameters["epochs"],
        device,
    )

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")

    # Saving the model
    if args.s:
        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
