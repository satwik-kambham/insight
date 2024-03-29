import gradio as gr
from huggingface_hub import hf_hub_download

import torch
import torch.nn as nn
from torchvision import transforms


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, set_stride=False):
        super().__init__()
        stride = 2 if in_channels != out_channels and set_stride else 1

        self.conv1 = nn.LazyConv2d(
            out_channels,
            kernel_size=3,
            padding="same" if stride == 1 else 1,
            stride=stride,
        )
        self.conv2 = nn.LazyConv2d(out_channels, kernel_size=3, padding="same")

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.LazyConv2d(out_channels, kernel_size=1, stride=stride),
                nn.LazyBatchNorm2d(),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.residual(x)
        out = self.relu(out)
        return out


class BottleneckResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, identity_mapping=False, set_stride=False
    ):
        super().__init__()
        stride = 2 if in_channels != out_channels and set_stride else 1

        self.conv1 = nn.LazyConv2d(
            out_channels,
            kernel_size=1,
            padding="same" if stride == 1 else 0,
            stride=stride,
        )
        self.conv2 = nn.LazyConv2d(out_channels, kernel_size=3, padding="same")
        self.conv3 = nn.LazyConv2d(out_channels * 4, kernel_size=1, padding="same")

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()

        self.relu = nn.ReLU()

        if in_channels != out_channels or not identity_mapping:
            self.residual = nn.Sequential(
                nn.LazyConv2d(out_channels * 4, kernel_size=1, stride=stride),
                nn.LazyBatchNorm2d(),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.residual(x)
        out = self.relu(out)
        return out


RESNET_18 = [2, 2, 2, 2]
RESNET_34 = [3, 4, 6, 3]
RESNET_50 = [3, 4, 6, 3]
RESNET_101 = [3, 4, 23, 3]
RESNET_152 = [3, 8, 36, 3]


class ResNet(nn.Module):
    def __init__(self, arch=RESNET_18, block="simple", num_classes=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(64, 64, arch[0], set_stride=False, block=block)
        self.conv3 = self._make_layer(64, 128, arch[1], block=block)
        self.conv4 = self._make_layer(128, 256, arch[2], block=block)
        self.conv5 = self._make_layer(256, 512, arch[3], block=block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(num_classes)

    def _make_layer(
        self, in_channels, out_channels, num_blocks, set_stride=True, block="simple"
    ):
        """Block is either 'simple' or 'bottleneck'"""
        layers = []
        for i in range(num_blocks):
            layers.append(
                SimpleResidualBlock(in_channels, out_channels, set_stride=set_stride)
                if block == "simple"
                else BottleneckResidualBlock(
                    in_channels if i == 0 else out_channels * 4,
                    out_channels,
                    set_stride=set_stride,
                )
            )
            set_stride = False
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(self.conv2(out))
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    def _init_weights(module):
        # Initlize weights with glorot uniform
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


class ImageClassifier:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model(checkpoint_path)
        self.transform = self.get_transform((244, 244))
        self.labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def load_model(self, checkpoint_path):
        classifier = ResNet(
            arch=RESNET_18,
            block="simple",
            num_classes=10,
        )
        classifier.load_state_dict(torch.load(checkpoint_path))
        classifier = classifier.cpu()
        classifier.eval()
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
        return {label: prob.item() for label, prob in zip(self.labels, probs)}

    def classify(self, input_image):
        return self.predict(input_image)


def classify(input_image):
    return classifier.classify(input_image)


checkpoint_path = hf_hub_download(
    repo_id="SatwikKambham/resnet18-cifar10",
    filename="model.pt",
)
classifier = ImageClassifier(checkpoint_path)
iface = gr.Interface(
    classify,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
    ],
    outputs=gr.Label(num_top_classes=3),
)

iface.launch()
