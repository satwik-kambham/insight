import torch
from tqdm import tqdm
import click

from ..data import dataloaders


def load_dataset(data_dir, dataset, size):
    if dataset == "OxfordIIITPet":
        return dataloaders.load_OxfordIIITPetDataset(data_dir, (size, size))
    elif dataset == "VOCSegmentation":
        return dataloaders.load_VOCSegmentationDataset(data_dir, False, (size, size))
    elif dataset == "VOCSegmentationSimple":
        return dataloaders.load_VOCSegmentationDataset(data_dir, True, (size, size))
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_class_distribution(dataset, data_dir, size):
    train_dataset, val_dataset, num_classes, class_weights = load_dataset(
        data_dir, dataset, size
    )
    class_frequencies = torch.zeros(num_classes)
    for i in tqdm(train_dataset):
        class_frequencies += torch.bincount(i[1].flatten(), minlength=num_classes)
    for i in tqdm(val_dataset):
        class_frequencies += torch.bincount(i[1].flatten(), minlength=num_classes)

    print(class_frequencies)

    # Calculate percentage of pixels in each class using class_frequencies
    class_percentage = class_frequencies / class_frequencies.sum()
    print(class_percentage)

    # Print in readable format
    for i in range(num_classes):
        print("Class {}: {}%".format(i, class_percentage[i] * 100))

    # Class weights
    # total_samples = class_frequencies.sum()
    # class_weights = total_samples / (num_classes * class_frequencies)
    print(class_weights)


def get_class_weights(train_dataset, val_dataset, num_classes):
    class_frequencies = torch.zeros(num_classes)
    for i in tqdm(train_dataset):
        class_frequencies += torch.bincount(i[1].flatten(), minlength=num_classes)
    for i in tqdm(val_dataset):
        class_frequencies += torch.bincount(i[1].flatten(), minlength=num_classes)

    # Class weights
    total_samples = class_frequencies.sum()
    class_weights = total_samples / (num_classes * class_frequencies)
    return torch.tensor(class_weights, dtype=torch.float32)


@click.command()
@click.option(
    "--dataset",
    default="VOCSegmentation",
    help="Dataset: VOCSegmentation, VOCSegmentationSimple, OxfordIIITPet",
)
@click.option("--data_dir", default="datasets", help="Path to data directory")
@click.option("--size", default=128, help="Input size")
def class_distribution_cli(dataset, data_dir, size):
    get_class_distribution(dataset, data_dir, size)


if __name__ == "__main__":
    class_distribution_cli()
