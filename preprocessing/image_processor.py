import cv2
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2

from utils.config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD
)


def get_validation_transform():

    return A.Compose([
        A.Resize(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE
        ),

        A.Normalize(
            mean=IMAGE_MEAN,
            std=IMAGE_STD
        ),

        ToTensorV2()
    ])


def load_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def preprocess_image(image):

    if image is None:
        raise ValueError("Input image cannot be None")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel RGB image"
        )

    original_image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    transform = get_validation_transform()

    transformed = transform(
        image=image
    )

    image_tensor = transformed["image"]

    return image_tensor, original_image


def preprocess_image_from_path(image_path):

    image = load_image(image_path)

    return preprocess_image(image)
