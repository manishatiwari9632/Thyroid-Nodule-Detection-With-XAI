
import base64
from io import BytesIO

from PIL import Image


def decode_image_data_url(image_data_url: str) -> Image.Image:
    """
    Decode a Base64 image data URL into a PIL Image.
    """

    if not image_data_url:
        raise ValueError("Image data is empty.")

    try:
        header, encoded_data = image_data_url.split(",", 1)

        if "base64" not in header:
            raise ValueError("Invalid Base64 image data URL.")

        image_bytes = base64.b64decode(
            encoded_data,
            validate=True,
        )

        image = Image.open(BytesIO(image_bytes))

        return image.convert("RGB")

    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(
            "Unable to decode the image data."
        ) from exc


def format_prediction_result(prediction: dict) -> str:
    """
    Format prediction details for frontend display.
    """

    predicted_class = prediction["predicted_class"]
    predicted_label = prediction["predicted_label"]

    benign_probability = prediction["benign_probability"] * 100
    malignant_probability = prediction["malignant_probability"] * 100
    confidence = prediction["confidence"] * 100
    threshold = prediction["threshold"]

    return (
        f"### Prediction Result: {predicted_class}\n\n"
        f"- **Predicted Label:** {predicted_label}\n"
        f"- **Confidence:** {confidence:.2f}%\n"
        f"- **Benign Probability:** {benign_probability:.2f}%\n"
        f"- **Malignant Probability:** {malignant_probability:.2f}%\n"
        f"- **Classification Threshold:** {threshold:.2f}"
    )
