import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchinfo import summary
import wandb

from LeNet import LeNet5

# Hyperparameters for training
hyperparameters = {
    "train_batch_size": 64,
    "val_batch_size": 64,
    "train_workers": 0,
    "val_workers": 0,
    "epochs": 10,
    "lr": 0.01,
}


def load_data(root):
    # Loading the dataset
    img_transform = transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        ]
    )
    label_transform = transforms.Compose([])
    dataset = datasets.Caltech256(
        root,
        transform=img_transform,
        target_transform=label_transform,
    )

    # Splitting the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

    # Creating the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters["train_batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=hyperparameters["train_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hyperparameters["val_batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=hyperparameters["val_workers"],
    )

    return train_loader, val_loader


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
    train_loss /= len(train_loader.dataset)
    return train_loss


def val_loop(model, loss_fn, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    return val_loss, val_accuracy


def train(model, loss_fn, optimizer, train_loader, val_loader, epochs, device):
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss = train_loop(model, loss_fn, optimizer, train_loader, device)
        val_loss, val_accuracy = val_loop(model, loss_fn, val_loader, device)
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
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
        "--device", type=str, default="cpu", help="device to use for training / testing"
    )

    args = parser.parse_args()

    hyperparameters["epochs"] = args.epochs
    hyperparameters["train_batch_size"] = args.batch_size
    hyperparameters["val_batch_size"] = args.batch_size
    hyperparameters["lr"] = args.learning_rate

    wandb.init(project="cnn-architectures", config=hyperparameters)

    device = torch.device(args.device)

    # Creating the model
    model = LeNet5(num_classes=257)
    test_input_size = (1, 3, 256, 256)
    test_input = torch.randn(test_input_size)
    test_output = model(test_input)

    # Printing the model summary
    model.eval()
    print(summary(model, input_size=test_input_size))

    # Defining the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"])

    # Loading the dataset
    train_loader, val_loader = load_data(args.data)

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
