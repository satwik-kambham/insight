from torchvision import transforms


def preprocess_image(img, img_shape):
    preprocess_transform = transforms.Compose(
        [
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return preprocess_transform(img)


def postprocess_mask(mask, img_shape):
    # Resize mask to original image shape
    mask = transforms.Resize(img_shape)(mask)
    return mask
