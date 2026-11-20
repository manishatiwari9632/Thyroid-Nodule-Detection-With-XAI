
import numpy as np
import torch

from preprocessing import preprocess_image
from utils.config import (
    CLASSIFICATION_THRESHOLD,
    CLASS_MAPPING,
    DEVICE
)


class PredictionService:

    def __init__(
        self,
        model,
        device=DEVICE,
        threshold=CLASSIFICATION_THRESHOLD
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Classification threshold must be between 0 and 1"
            )

        self.model = model
        self.device = torch.device(device)
        self.threshold = threshold

        self.model.to(self.device)
        self.model.eval()

    def predict_preprocessed(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray
    ) -> dict:
        """
        Predict using an already-preprocessed image.

        Args:
            image_tensor:
                Batched tensor with shape [1, C, H, W].

            original_image:
                Original RGB uint8 image.

        Returns:
            Prediction results and original image.
        """

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(
                "image_tensor must be a PyTorch tensor"
            )

        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError(
                "Expected image_tensor with shape [1, C, H, W]"
            )

        if not isinstance(original_image, np.ndarray):
            raise TypeError(
                "original_image must be a NumPy array"
            )

        if (
            original_image.ndim != 3
            or original_image.shape[2] != 3
        ):
            raise ValueError(
                "Expected original_image with shape [H, W, 3]"
            )

        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            output = self.model(image_tensor)

            if output.shape != (1, 1):
                raise RuntimeError(
                    "Expected model output with shape [1, 1]"
                )

            malignant_probability = torch.sigmoid(
                output
            ).item()

        benign_probability = 1.0 - malignant_probability

        predicted_label = int(
            malignant_probability >= self.threshold
        )

        predicted_class = CLASS_MAPPING[predicted_label]

        confidence = (
            malignant_probability
            if predicted_label == 1
            else benign_probability
        )

        return {
            "predicted_label": predicted_label,
            "predicted_class": predicted_class,
            "malignant_probability": malignant_probability,
            "benign_probability": benign_probability,
            "confidence": confidence,
            "threshold": self.threshold,
            "original_image": original_image
        }

    def predict(self, image: np.ndarray) -> dict:
        """
        Preprocess an RGB image and perform prediction.

        Retains the original PredictionService interface.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Expected image to be a NumPy array"
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                "Expected an RGB image with shape [H, W, 3]"
            )

        image_tensor, original_image = preprocess_image(image)

        image_tensor = image_tensor.unsqueeze(0)

        return self.predict_preprocessed(
            image_tensor=image_tensor,
            original_image=original_image
        )
