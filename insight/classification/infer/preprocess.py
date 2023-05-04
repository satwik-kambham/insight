from torchvision import transforms


def preprocess_image(img, img_shape):
    preprocess_transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
        ]
    )

    return preprocess_transform(img)
