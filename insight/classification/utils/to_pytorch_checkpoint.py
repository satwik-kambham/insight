import sys
import torch

from ..models import classifier


def to_pytorch_checkpoint():
    checkpoint_path = sys.argv[1]
    checkpoint_folder = "/".join(checkpoint_path.split("/")[:-1])
    model = classifier.Classifier.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model = model.cpu()
    model.eval()
    model.freeze()

    torch.save(model.classifier.state_dict(), checkpoint_folder + "/model.pt")


if __name__ == "__main__":
    to_pytorch_checkpoint()
