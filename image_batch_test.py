
import numpy as np

from services.batch_inference_service import BatchInferenceService


class MockInferenceService:

    def predict_with_explanation(self, image):

        return {
            "predicted_label": "Benign",
            "predicted_class": 0,
            "benign_probability": 0.8,
            "malignant_probability": 0.2,
            "confidence": 0.8,
            "threshold": 0.5,
            "original_image": image,
            "heatmap": np.zeros((224, 224)),
            "overlay": image,
            "explained_class": 0,
            "explained_label": "Benign",
        }


def test_process_multiple_uploaded_images():

    inference_service = MockInferenceService()

    batch_service = BatchInferenceService.__new__(
        BatchInferenceService
    )

    batch_service.inference_service = inference_service

    images = [
        ("image1.jpg", np.zeros((224, 224, 3), dtype=np.uint8)),
        ("image2.png", np.zeros((224, 224, 3), dtype=np.uint8)),
    ]

    result = batch_service.process_images(images)

    assert result["total_images"] == 2
    assert result["successful_count"] == 2
    assert result["failed_count"] == 0
    assert len(result["results"]) == 2
    assert len(result["failed_images"]) == 0


def test_process_uploaded_image_with_invalid_input():

    inference_service = MockInferenceService()

    batch_service = BatchInferenceService.__new__(
        BatchInferenceService
    )

    batch_service.inference_service = inference_service

    images = [
        ("valid.jpg", np.zeros((224, 224, 3), dtype=np.uint8)),
        ("invalid.jpg", "invalid image data"),
    ]

    result = batch_service.process_images(images)

    assert result["total_images"] == 2
    assert result["successful_count"] == 1
    assert result["failed_count"] == 1

    assert result["failed_images"][0]["path"] == "invalid.jpg"

    assert len(result["results"]) == 1
