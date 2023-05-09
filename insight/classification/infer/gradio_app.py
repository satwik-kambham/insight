import gradio as gr

import torch
from torchvision import transforms

from ..models.classifier import Classifier


class ImageClassifier:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model(checkpoint_path)
        self.transform = self.get_transform(self.model.hparams.img_shape)

    def load_model(self, checkpoint_path):
        classifier = Classifier.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
        )
        classifier = classifier.cpu()
        classifier.eval()
        classifier.freeze()
        return classifier

    def get_transform(self, img_shape):
        preprocess_transform = transforms.Compose(
            [
                transforms.Resize(img_shape),
                transforms.ToTensor(),
            ]
        )
        return preprocess_transform

    def predict(self, image):
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = logits.softmax(dim=1)[0]
        return {label: prob.item() for label, prob in zip(self.model.labels, probs)}

    def classify(self, input_image, checkpoint_file):
        self.checkpoint_path = checkpoint_file.name
        self.model = self.load_model(checkpoint_file.name)
        return self.predict(input_image)


def classify(input_image, checkpoint_file):
    classifier = ImageClassifier(checkpoint_file.name)
    print("Categories:", classifier.model.labels)
    return classifier.classify(input_image, checkpoint_file)


if __name__ == "__main__":
    iface = gr.Interface(
        classify,
        inputs=[
            gr.Image(label="Input Image", type="pil"),
            gr.File(label="Model Checkpoint"),
        ],
        outputs=gr.Label(num_top_classes=3),
    )

    iface.launch()
