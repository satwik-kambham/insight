from lightning.pytorch.cli import LightningCLI

from dataset import TREC, RNN_DataModule
from classifier import Classifier


def cli_main():
    cli = LightningCLI(Classifier, RNN_DataModule)


if __name__ == "__main__":
    dataset = TREC()
    print("vocab_size:", dataset.tokenizer.count)
    cli_main()
