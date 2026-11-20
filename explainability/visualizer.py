
import cv2
import numpy as np


class GradCAMVisualizer:
    """
    Creates RGB Grad-CAM heatmap overlays using OpenCV.
    """

    def __init__(self, alpha: float = 0.4):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "Alpha must be between 0 and 1"
            )

        self.alpha = alpha

    def generate_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Resize, colorize and overlay a Grad-CAM heatmap.

        Args:
            original_image:
                Original RGB image with shape [H, W, 3],
                expected to have dtype uint8.

            heatmap:
                Normalized Grad-CAM heatmap with shape [H, W].

        Returns:
            RGB overlay image with shape [H, W, 3]
            and dtype uint8.
        """

        if not isinstance(original_image, np.ndarray):
            raise TypeError(
                "original_image must be a NumPy array"
            )

        if not isinstance(heatmap, np.ndarray):
            raise TypeError(
                "heatmap must be a NumPy array"
            )

        if (
            original_image.ndim != 3
            or original_image.shape[2] != 3
            or original_image.shape[0] == 0
            or original_image.shape[1] == 0
        ):
            raise ValueError(
                "Expected a non-empty RGB image with shape [H, W, 3]"
            )

        if original_image.dtype != np.uint8:
            raise TypeError(
                "original_image must have dtype uint8"
            )

        if (
            heatmap.ndim != 2
            or heatmap.shape[0] == 0
            or heatmap.shape[1] == 0
        ):
            raise ValueError(
                "Expected a non-empty 2D heatmap"
            )

        if not np.isfinite(heatmap).all():
            raise ValueError(
                "Heatmap contains non-finite values"
            )

        # Resize heatmap to match the original image dimensions.
        height, width = original_image.shape[:2]

        resized_heatmap = cv2.resize(
            heatmap.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert normalized heatmap to uint8.
        heatmap_uint8 = (
            np.clip(resized_heatmap, 0.0, 1.0) * 255
        ).astype(np.uint8)

        # Apply JET colormap.
        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # OpenCV produces BGR; convert to RGB.
        heatmap_colored = cv2.cvtColor(
            heatmap_colored,
            cv2.COLOR_BGR2RGB
        )

        # Blend original image and colored heatmap.
        overlay = cv2.addWeighted(
            original_image,
            1.0 - self.alpha,
            heatmap_colored,
            self.alpha,
            0
        )

        return overlay
