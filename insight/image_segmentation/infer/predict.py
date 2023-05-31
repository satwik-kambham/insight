import click

from PIL import Image

from ..models.unet import UNetModule
from .process import preprocess_image, postprocess_mask
from ..utils.mask import generate_mask


@click.command()
@click.option("--ckpt_file", type=str, help="Path to checkpoint file")
@click.option("--img", type=str, help="Path to image")
@click.option("--output", type=str, help="Path to output mask", default="mask.png")
def infer(ckpt_file, img, output):
    segmenter = UNetModule.load_from_checkpoint(
        checkpoint_path=ckpt_file,
    )
    segmenter = segmenter.cpu()

    print(segmenter.hparams)

    img = Image.open(img)
    img_shape = img.size
    img = preprocess_image(
        img,
        (
            segmenter.hparams.inp_size,
            segmenter.hparams.inp_size,
        ),
    )

    segmenter.eval()
    segmenter.freeze()

    pred = segmenter(img.unsqueeze(0))

    mask = generate_mask(pred, num_classes=segmenter.hparams.num_classes)
    mask = postprocess_mask(mask, (img_shape[1], img_shape[0]))
    mask.save(output)


if __name__ == "__main__":
    infer()
