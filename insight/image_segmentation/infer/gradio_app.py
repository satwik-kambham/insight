import gradio as gr

import torch
from torchvision import transforms

from ..models.unet import UNetModule
from .process import postprocess_mask
from ..utils.mask import generate_mask


class SemanticSegmenter:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model(checkpoint_path)
        self.transform = self.get_transform(
            (
                self.model.hparams.inp_size,
                self.model.hparams.inp_size,
            )
        )

    def load_model(self, checkpoint_path):
        segmenter = UNetModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
        )
        segmenter = segmenter.cpu()
        segmenter.eval()
        segmenter.freeze()
        return segmenter

    def get_transform(self, img_shape):
        preprocess_transform = transforms.Compose(
            [
                transforms.Resize(img_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return preprocess_transform

    def predict(self, image, num_classes):
        img_shape = image.size
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(image_tensor)
            mask = generate_mask(pred, num_classes=num_classes)
            mask = postprocess_mask(mask, (img_shape[1], img_shape[0]))
        return mask

    def segment(self, input_image, checkpoint_file):
        self.checkpoint_path = checkpoint_file.name
        self.model = self.load_model(checkpoint_file.name)
        return self.predict(input_image, self.model.hparams.num_classes)


def segment(input_image, checkpoint_file):
    segmenter = SemanticSegmenter(checkpoint_file.name)
    return segmenter.segment(input_image, checkpoint_file)


if __name__ == "__main__":
    iface = gr.Interface(
        segment,
        inputs=[
            gr.Image(label="Input Image", type="pil"),
            gr.File(label="Model Checkpoint"),
        ],
        outputs=gr.Image(label="Segmented Mask", type="pil"),
    )

    iface.launch()
