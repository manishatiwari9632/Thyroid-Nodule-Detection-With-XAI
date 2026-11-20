import base64
import logging
from pathlib import Path

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.schemas import (
    HealthResponse,
    SinglePredictionResponse,
    BatchPredictionResponse,
)

from utils.config import (
    MAX_UPLOAD_SIZE_MB,
    MAX_BATCH_SIZE_MB,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_BATCH_SIZE_BYTES,
)

from utils.image_collection import SUPPORTED_IMAGE_EXTENSIONS


router = APIRouter()

logger = logging.getLogger(__name__)

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode an RGB image as a Base64 PNG data URL.
    """

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, encoded_image = cv2.imencode(".png", image_bgr)

    if not success:
        raise ValueError("Failed to encode image.")

    image_base64 = base64.b64encode(encoded_image).decode("utf-8")

    return f"data:image/png;base64,{image_base64}"


@router.get(
    "/health",
    tags=["Health"],
    response_model=HealthResponse,
)
async def health_check(request: Request):

    model_loaded = getattr(request.app.state, "model", None) is not None

    inference_service_loaded = (
        getattr(request.app.state, "inference_service", None) is not None
    )

    batch_service_loaded = (
        getattr(request.app.state, "batch_inference_service", None) is not None
    )

    services_ready = (
        model_loaded
        and inference_service_loaded
        and batch_service_loaded
    )

    return {
        "success": services_ready,
        "message": (
            "Thyroid Detection API is healthy."
            if services_ready
            else "Thyroid Detection API is not fully initialized."
        ),
        "data": {
            "model_loaded": model_loaded,
            "inference_service_loaded": inference_service_loaded,
            "batch_inference_service_loaded": batch_service_loaded,
        },
    }


@router.post(
    "/predict",
    tags=["Prediction"],
    response_model=SinglePredictionResponse,
)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Perform thyroid ultrasound image classification
    and generate Grad-CAM explanation.
    """

    # 1. Validate filename and extension.
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {file_extension}",
        )

    # 2. Read uploaded image with a size limit.
    image_bytes = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image size exceeds the 10 MB upload limit.",
        )

    # 3. Decode and validate actual image content.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    # Convert BGR to RGB because InferenceService expects RGB.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 4. Retrieve initialized inference service.
    inference_service = getattr(
        request.app.state,
        "inference_service",
        None,
    )

    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service is not available.",
        )

    # 5. Run prediction and Grad-CAM explanation.
    try:
        result = await run_in_threadpool(
            inference_service.predict_with_explanation,
            image_rgb,
        )

        # 6. Convert visualization arrays into Base64 PNGs.
        original_image = result["original_image"]
        heatmap = result["heatmap"]
        overlay = result["overlay"]

        height, width = original_image.shape[:2]

        # Resize heatmap to match the original image dimensions.
        heatmap_resized = cv2.resize(
            heatmap,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

        heatmap_uint8 = (
            np.clip(heatmap_resized, 0, 1) * 255
        ).astype(np.uint8)

        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_colored,
            cv2.COLOR_BGR2RGB,
        )

        heatmap_base64 = encode_image_to_base64(heatmap_rgb)
        overlay_base64 = encode_image_to_base64(overlay)

        # 7. Return JSON-compatible response.
        return {
            "success": True,
            "message": "Prediction completed successfully.",
            "data": {
                "filename": file.filename,
                "prediction": {
                    "predicted_label": result["predicted_label"],
                    "predicted_class": result["predicted_class"],
                    "benign_probability": result["benign_probability"],
                    "malignant_probability": result["malignant_probability"],
                    "confidence": result["confidence"],
                    "threshold": result["threshold"],
                },
                "explanation": {
                    "explained_class": result["explained_class"],
                    "explained_label": result["explained_label"],
                    "heatmap": heatmap_base64,
                    "overlay": overlay_base64,
                },
            },
        }

    except Exception:
        logger.exception(
            "Prediction failed for uploaded image: %s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image.",
        )

    finally:
        await file.close()


@router.post(
    "/predict/batch",
    tags=["Prediction"],
    response_model=BatchPredictionResponse,
)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
):
    """
    Perform batch thyroid ultrasound classification
    and Grad-CAM explanation.

    Supports:
        1. Multiple individually selected images.
        2. Images uploaded from a selected folder.

    Folder uploads process root-level images only.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No images were uploaded.",
        )

    batch_service = getattr(
        request.app.state,
        "batch_inference_service",
        None,
    )

    if batch_service is None:
        raise HTTPException(
            status_code=503,
            detail="Batch inference service is not available.",
        )

    uploaded_images = []
    failed_images = []
    total_upload_size = 0

    try:

        # --------------------------------------------------
        # 1. Validate and decode uploaded images
        # --------------------------------------------------

        for uploaded_file in files:

            filename = uploaded_file.filename or "unknown"

            try:

                # Normalize browser-provided relative paths.
                normalized_path = filename.replace("\\", "/")

                if normalized_path.startswith("/"):
                    raise ValueError("Absolute file paths are not allowed.")

                path_parts = [
                    part
                    for part in normalized_path.split("/")
                    if part not in ("", ".")
                ]

                if (
                    not path_parts
                    or any(part == ".." for part in path_parts)
                ):
                    raise ValueError("Invalid image filename or path.")

                # A folder upload may contain:
                # selected_folder/image.jpg
                #
                # Nested subfolders are not supported.
                if len(path_parts) > 2:
                    raise ValueError(
                        "Images inside nested subfolders are not supported."
                    )

                image_filename = path_parts[-1]

                file_extension = Path(image_filename).suffix.lower()

                if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported image format: {file_extension}"
                    )

                # Read with individual file-size limit.
                image_bytes = await uploaded_file.read(
                    MAX_UPLOAD_SIZE_BYTES + 1
                )

                if not image_bytes:
                    raise ValueError("Uploaded file is empty.")

                if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
                    raise ValueError(
                        "Image size exceeds the 10 MB upload limit."
                    )

                total_upload_size += len(image_bytes)

                if total_upload_size > MAX_BATCH_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Total batch upload size exceeds 50 MB.",
                    )

                # Decode image content.
                image_array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8,
                )

                image_bgr = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR,
                )

                if image_bgr is None:
                    raise ValueError(
                        "Uploaded file is not a valid image."
                    )

                # Convert BGR to RGB.
                image_rgb = cv2.cvtColor(
                    image_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                uploaded_images.append(
                    (image_filename, image_rgb)
                )

            except HTTPException:
                raise

            except Exception as exc:

                failed_images.append({
                    "path": filename,
                    "reason": str(exc),
                })

            finally:
                await uploaded_file.close()

        # --------------------------------------------------
        # 2. Run batch inference
        # --------------------------------------------------

        batch_result = await run_in_threadpool(
            batch_service.process_images,
            uploaded_images,
        )

        # --------------------------------------------------
        # 3. Prepare JSON-compatible results
        # --------------------------------------------------

        formatted_results = []

        for result in batch_result["results"]:

            try:

                heatmap = result["heatmap"]
                overlay = result["overlay"]

                original_image = result["original_image"]

                height, width = original_image.shape[:2]

                # Resize heatmap to original image dimensions.
                heatmap_resized = cv2.resize(
                    heatmap,
                    (width, height),
                    interpolation=cv2.INTER_CUBIC,
                )

                heatmap_uint8 = (
                    np.clip(heatmap_resized, 0, 1) * 255
                ).astype(np.uint8)

                heatmap_colored = cv2.applyColorMap(
                    heatmap_uint8,
                    cv2.COLORMAP_JET,
                )

                heatmap_rgb = cv2.cvtColor(
                    heatmap_colored,
                    cv2.COLOR_BGR2RGB,
                )

                formatted_results.append({
                    "filename": result["image_path"],
                    "prediction": {
                        "predicted_label": result["predicted_label"],
                        "predicted_class": result["predicted_class"],
                        "benign_probability": result["benign_probability"],
                        "malignant_probability": result["malignant_probability"],
                        "confidence": result["confidence"],
                        "threshold": result["threshold"],
                    },
                    "explanation": {
                        "explained_class": result["explained_class"],
                        "explained_label": result["explained_label"],
                        "heatmap": encode_image_to_base64(heatmap_rgb),
                        "overlay": encode_image_to_base64(overlay),
                    },
                })

            except Exception as exc:

                failed_images.append({
                    "path": result.get("image_path", "unknown"),
                    "reason": str(exc),
                })

        # --------------------------------------------------
        # 4. Calculate final batch summary
        # --------------------------------------------------

        all_failures = (
            batch_result["failed_images"] + failed_images
        )

        total_images = len(files)

        successful_count = len(formatted_results)
        failed_count = len(all_failures)

        return {
            "success": True,
            "message": "Batch prediction completed.",
            "data": {
                "summary": {
                    "total_images": total_images,
                    "successful_count": successful_count,
                    "failed_count": failed_count,
                },
                "results": formatted_results,
                "failed_images": all_failures,
            },
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception("Batch prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the batch.",
        )
