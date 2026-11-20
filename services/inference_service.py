
import numpy as np

from preprocessing.image_processor import preprocess_image
from services.prediction_service import PredictionService
from services.explanation_service import ExplanationService
from utils.config import (
    CLASSIFICATION_THRESHOLD,
    DEVICE
)


class InferenceService:
    """
    Coordinates image preprocessing, prediction and Grad-CAM
    explanation in a single inference workflow.
    """

    def __init__(
        self,
        model,
        device=DEVICE,
        threshold=CLASSIFICATION_THRESHOLD,
        alpha=0.4
    ):
        self.prediction_service = PredictionService(
            model=model,
            device=device,
            threshold=threshold
        )

        self.explanation_service = ExplanationService(
            model=model,
            device=device,
            alpha=alpha
        )

    def predict_with_explanation(
        self,
        image: np.ndarray
    ) -> dict:
        """
        Perform prediction and generate a Grad-CAM explanation.

        Args:
            image:
                RGB NumPy image.

        Returns:
            Combined prediction and explanation results.
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

        # Preprocess the image only once.
        image_tensor, original_image = preprocess_image(image)

        # Add batch dimension for model inference and Grad-CAM.
        image_tensor = image_tensor.unsqueeze(0)

        # Perform classification using the preprocessed tensor.
        prediction = self.prediction_service.predict_preprocessed(
            image_tensor=image_tensor,
            original_image=original_image
        )

        # Generate Grad-CAM for the predicted class.
        explanation = self.explanation_service.generate_explanation(
            image_tensor=image_tensor,
            original_image=original_image,
            predicted_class=prediction["predicted_label"]
        )

        # Combine prediction and explanation results.
        return {
            **prediction,
            **explanation
        }
