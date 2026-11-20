
from pathlib import Path

import cv2
import numpy as np

from services.inference_service import InferenceService
from utils.image_collection import (
    collect_image_paths,
    collect_images_from_folder,
)


class BatchInferenceService:
    """
    Handles batch inference for multiple thyroid ultrasound images.

    Supports:
        1. Individually selected image paths.
        2. Images collected from a folder's root directory.
        3. Uploaded images represented as RGB NumPy arrays.
    """

    def __init__(self, inference_service: InferenceService):
        """
        Args:
            inference_service: Initialized InferenceService instance.
        """

        if not isinstance(inference_service, InferenceService):
            raise TypeError(
                "inference_service must be an InferenceService instance."
            )

        self.inference_service = inference_service

    def _process_image_array(self, image, image_identifier):
        """
        Process an already-loaded RGB image.

        Args:
            image: RGB NumPy image.
            image_identifier: Image filename or path.

        Returns:
            Dictionary containing inference results.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Image must be an RGB image with three channels."
            )

        result = self.inference_service.predict_with_explanation(image)

        result["image_path"] = str(image_identifier)

        return result

    def _process_single_image(self, image_path):
        """
        Load and process a single image from its path.
        """

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(
                "Unable to read image. The file may be corrupted "
                "or contain unsupported image data."
            )

        # Convert OpenCV BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self._process_image_array(
            image=image,
            image_identifier=image_path,
        )

    def _process_collected_images(self, collection_result):
        """
        Process all valid images and collect individual results.

        Invalid or failed images do not stop the batch.
        """

        successful_results = []
        failed_images = []

        valid_images = collection_result["valid_images"]
        invalid_images = collection_result["invalid_images"]

        # Include collection failures in the final report.
        failed_images.extend(invalid_images)

        for image_path in valid_images:

            try:
                result = self._process_single_image(image_path)
                successful_results.append(result)

            except Exception as exc:
                failed_images.append({
                    "path": str(image_path),
                    "reason": str(exc),
                })

        total_images = len(valid_images) + len(invalid_images)

        return {
            "total_images": total_images,
            "successful_count": len(successful_results),
            "failed_count": len(failed_images),
            "results": successful_results,
            "failed_images": failed_images,
        }

    def process_image_paths(self, image_paths):
        """
        Process individually selected image paths.

        Args:
            image_paths: List or tuple of image paths.

        Returns:
            Batch inference results.
        """

        collection_result = collect_image_paths(image_paths)

        return self._process_collected_images(collection_result)

    def process_folder(self, folder_path):
        """
        Process supported images directly from a folder's root.

        Nested subdirectories are ignored.

        Args:
            folder_path: Selected folder path.

        Returns:
            Batch inference results.
        """

        collection_result = collect_images_from_folder(folder_path)

        return self._process_collected_images(collection_result)

    def process_images(self, images):
        """
        Process uploaded images represented as RGB NumPy arrays.

        Args:
            images: List of tuples containing:
                (filename, RGB NumPy image)

        Returns:
            Batch inference results.
        """

        successful_results = []
        failed_images = []

        for filename, image in images:

            try:
                result = self._process_image_array(
                    image=image,
                    image_identifier=filename,
                )

                successful_results.append(result)

            except Exception as exc:
                failed_images.append({
                    "path": str(filename),
                    "reason": str(exc),
                })

        return {
            "total_images": len(images),
            "successful_count": len(successful_results),
            "failed_count": len(failed_images),
            "results": successful_results,
            "failed_images": failed_images,
        }
