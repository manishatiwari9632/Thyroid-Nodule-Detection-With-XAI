
import numpy as np
import torch

from explainability.gradcam import GradCAM
from explainability.visualizer import GradCAMVisualizer
from utils.config import CLASS_MAPPING, DEVICE


class ExplanationService:
    """
    Coordinates Grad-CAM generation and heatmap visualization.
    """

    def __init__(
        self,
        model,
        device=DEVICE,
        alpha: float = 0.4
    ):
        self.device = torch.device(device)

        # Initialize these components once and reuse them.
        self.gradcam = GradCAM(
            model=model,
            device=self.device
        )

        self.visualizer = GradCAMVisualizer(
            alpha=alpha
        )

    def generate_explanation(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
        predicted_class: int
    ) -> dict:
        """
        Generate a class-specific Grad-CAM explanation.

        Args:
            image_tensor:
                Preprocessed image tensor with shape [1, C, H, W].

            original_image:
                Original RGB image as a uint8 NumPy array.

            predicted_class:
                Predicted class index:
                    0 -> Benign
                    1 -> Malignant

        Returns:
            Dictionary containing the explained class,
            raw heatmap and RGB overlay.
        """

        if (
            isinstance(predicted_class, bool)
            or not isinstance(predicted_class, (int, np.integer))
            or predicted_class not in CLASS_MAPPING
        ):
            raise ValueError(
                "predicted_class must be 0 (Benign) or 1 (Malignant)"
            )

        # Generate Grad-CAM for the predicted class.
        heatmap = self.gradcam.generate_heatmap(
            img_tensor=image_tensor,
            pred_index=int(predicted_class)
        )

        # Blend the heatmap with the original RGB image.
        overlay = self.visualizer.generate_overlay(
            original_image=original_image,
            heatmap=heatmap
        )

        return {
            "explained_class": int(predicted_class),
            "explained_label": CLASS_MAPPING[int(predicted_class)],
            "heatmap": heatmap,
            "overlay": overlay
        }
