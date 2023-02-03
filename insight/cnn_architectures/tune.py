import argparse

import torch
import torch.nn as nn
import torch.optim as optim

# import wandb

import optuna
from optuna.trial import TrialState

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


def train(model, loss_fn, optimizer, train_loader, val_loader, device):
    model = model.to(device)
    train_loss = train_loop(model, loss_fn, optimizer, train_loader, device)
    val_loss, val_accuracy = val_loop(model, loss_fn, val_loader, device)
    return train_loss, val_loss, val_accuracy


def main():
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
        "--device", type=str, default="cpu", help="device to use for training / testing"
    )

    parser.add_argument(
        "--nightly", action="store_true", help="Use pytorch 2.0 to compile model"
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

    # wandb.init(project="cnn-architectures", config=hyperparameters)

    def objective(trail):
        # Loading the dataset
        train_loader, val_loader, num_classes, img_shape, num_channels = load_data(
            args.data,
            hyperparameters,
        )

        hyperparameters["img_shape"] = img_shape

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
        elif hyperparameters["architecture"] == "GoogLeNet":
            model = GoogLeNet(num_classes=num_classes)
        elif hyperparameters["architecture"] == "ResNet18":
            model = ResNet(RESNET_18, num_classes=num_classes)
        elif hyperparameters["architecture"] == "ResNet34":
            model = ResNet(RESNET_34, num_classes=num_classes)
        elif hyperparameters["architecture"] == "ResNet50":
            model = ResNet(RESNET_50, block="bottleneck", num_classes=num_classes)
        elif hyperparameters["architecture"] == "ResNet101":
            model = ResNet(RESNET_101, block="bottleneck", num_classes=num_classes)
        elif hyperparameters["architecture"] == "ResNet152":
            model = ResNet(RESNET_152, block="bottleneck", num_classes=num_classes)
        else:
            raise NotImplementedError
        test_input_size = (1, num_channels, *hyperparameters["img_shape"])
        test_input = torch.randn(test_input_size)
        _ = model(test_input)

        if hasattr(model, "_init_weights"):
            model._init_weights()

        if args.nightly:
            model = torch.compile(model)

        # Defining the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        lr = trail.suggest_loguniform("lr", 1e-4, 1e-1)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(hyperparameters["epochs"]):
            # Training the model
            train_loss, val_loss, val_accuracy = train(
                model,
                loss_fn,
                optimizer,
                train_loader,
                val_loader,
                device,
            )

            trail.report(val_accuracy, epoch)

            if trail.should_prune():
                raise optuna.TrialPruned()

        return val_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
