import click

from PIL import Image

from ..models.classifier import Classifier

from .preprocess import preprocess_image


@click.command()
@click.option("--ckpt_file", type=str, help="Path to checkpoint file")
@click.option("--img", type=str, help="Path to image")
def infer(ckpt_file, img):
    classifier = Classifier.load_from_checkpoint(
        checkpoint_path=ckpt_file,
    )
    classifier = classifier.cpu()

    print(classifier.hparams)

    img = Image.open(img)
    img = preprocess_image(img, classifier.hparams.img_shape)

    classifier.eval()
    classifier.freeze()

    logits = classifier(img.unsqueeze(0))
    print(logits)

    probs = logits.softmax(dim=1)
    print(probs)

    # Print category along with probability ordered from highest to lowest
    for i, (prob, label) in enumerate(
        sorted(zip(probs.squeeze().tolist(), classifier.labels), reverse=True)
    ):
        print(f"{i + 1}. {label}: {prob * 100:.2f}%")


if __name__ == "__main__":
    infer()
