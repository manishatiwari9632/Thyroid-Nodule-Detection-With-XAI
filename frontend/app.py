
import gradio as gr

from frontend.api_client import (
    ThyroidAPIClient,
    APIClientError,
)

from frontend.config import (
    FRONTEND_HOST,
    FRONTEND_PORT,
)

from frontend.utils import (
    decode_image_data_url,
    format_prediction_result,
)


# ==========================================================
# API CLIENT
# ==========================================================

api_client = ThyroidAPIClient()


# ==========================================================
# BACKEND HEALTH CHECK
# ==========================================================

def check_backend():

    try:
        response = api_client.health_check()

        if response["success"]:
            return "🟢 Backend connected successfully."

        return "🟠 Backend is not fully initialized."

    except APIClientError as exc:
        return f"🔴 Backend connection failed: {exc}"


# ==========================================================
# SINGLE IMAGE PREDICTION
# ==========================================================

def predict_single_image(image_path):

    if not image_path:
        raise gr.Error("Please upload a thyroid ultrasound image.")

    try:

        response = api_client.predict_image(image_path)

        if not response["success"]:
            raise gr.Error(response["message"])

        data = response["data"]

        prediction = data["prediction"]
        explanation = data["explanation"]

        result_markdown = format_prediction_result(prediction)

        heatmap = decode_image_data_url(
            explanation["heatmap"]
        )

        overlay = decode_image_data_url(
            explanation["overlay"]
        )

        return (
            result_markdown,
            heatmap,
            overlay,
        )

    except APIClientError as exc:
        raise gr.Error(str(exc))

    except (KeyError, ValueError) as exc:
        raise gr.Error(
            f"Unable to process prediction response: {exc}"
        )

# ==========================================================
# BATCH INFERENCE
# ==========================================================

def predict_batch_images(multiple_files, folder_files):
    image_paths = []

    # Collect individually selected images.
    if multiple_files:
        if isinstance(multiple_files, str):
            image_paths.append(multiple_files)
        else:
            image_paths.extend(multiple_files)

    # Collect images selected through folder upload.
    if folder_files:
        if isinstance(folder_files, str):
            image_paths.append(folder_files)
        else:
            image_paths.extend(folder_files)

    # Remove duplicate paths while preserving selection order.
    image_paths = list(dict.fromkeys(image_paths))

    if not image_paths:
        raise gr.Error("Please select images or upload a folder.")

    try:
        response = api_client.predict_batch(image_paths)

        if not response["success"]:
            raise gr.Error(response["message"])

        data = response["data"]
        results = data.get("results", [])
        failed_images = data.get("failed_images", [])

        rows = []

        for item in results:
            prediction = item.get("prediction", item)

            rows.append([
                item.get("filename", "Unknown"),
                prediction.get("predicted_class", "Unknown"),
                f"{prediction.get('confidence', 0) * 100:.2f}%",
                f"{prediction.get('benign_probability', 0) * 100:.2f}%",
                f"{prediction.get('malignant_probability', 0) * 100:.2f}%",
            ])

        successful_count = len(results)
        failed_count = len(failed_images)
        total_count = successful_count + failed_count

        summary = (
            "## Batch Prediction Summary\n\n"
            f"- **Total Images:** {total_count}\n"
            f"- **Successful Predictions:** {successful_count}\n"
            f"- **Failed Images:** {failed_count}"
        )

        failure_details = "### Failed Images\n\n"

        if failed_images:
            for item in failed_images:
                filename = item.get("image_path", item.get("filename", "Unknown"))
                reason = item.get("error", item.get("message", "Unknown error"))
                failure_details += f"- **{filename}:** {reason}\n"
        else:
            failure_details += "No failed images."

        return summary, rows, failure_details

    except APIClientError as exc:
        raise gr.Error(str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        raise gr.Error(f"Unable to process batch prediction response: {exc}")

# ==========================================================
# GRADIO INTERFACE
# ==========================================================

with gr.Blocks(
    title="Thyroid Nodule Detection",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # Thyroid Nodule Detection

        ### FibonacciNet Classification with Grad-CAM Explainability

        Upload a thyroid ultrasound image to obtain a prediction
        and visualize the model's attention regions.
        """
    )

    # Backend status
    with gr.Row():
        backend_status = gr.Markdown(
            value=check_backend()
        )

        refresh_button = gr.Button(
            "Check Backend",
            size="sm",
        )

    refresh_button.click(
        fn=check_backend,
        outputs=backend_status,
    )

    gr.Markdown("---")

    # Single image prediction
    gr.Markdown("## Single Image Prediction")

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="filepath",
                label="Upload Thyroid Ultrasound Image",
                sources=["upload"],
            )

            predict_button = gr.Button(
                "Predict",
                variant="primary",
            )

        with gr.Column(scale=1):

            prediction_output = gr.Markdown(
                label="Prediction Result"
            )

    # Grad-CAM visualizations
    gr.Markdown("## Grad-CAM Explainability")

    with gr.Row():

        heatmap_output = gr.Image(
            label="Grad-CAM Heatmap",
            type="pil",
        )

        overlay_output = gr.Image(
            label="Grad-CAM Overlay",
            type="pil",
        )

    predict_button.click(
        fn=predict_single_image,
        inputs=image_input,
        outputs=[
            prediction_output,
            heatmap_output,
            overlay_output,
        ],
    )


    # ==========================================================
    # BATCH IMAGE PREDICTION
    # ==========================================================

    gr.Markdown("---")
    gr.Markdown("## Batch Image Prediction")
    gr.Markdown(
        "Select multiple ultrasound images or upload a folder "
        "containing images. Only images in the selected folder's "
        "root directory will be processed."
    )

    with gr.Row():
        with gr.Column():
            multiple_images_input = gr.File(
                label="Select Multiple Images",
                file_count="multiple",
                file_types=[
                    ".jpg", ".jpeg", ".png", ".bmp",
                    ".tif", ".tiff", ".webp",
                ],
                type="filepath",
            )

            folder_images_input = gr.File(
                label="Select Image Folder",
                file_count="directory",
                file_types=[
                    ".jpg", ".jpeg", ".png", ".bmp",
                    ".tif", ".tiff", ".webp",
                ],
                type="filepath",
            )

            batch_predict_button = gr.Button(
                "Run Batch Prediction",
                variant="primary",
            )

    batch_summary_output = gr.Markdown()

    batch_results_output = gr.Dataframe(
        headers=[
            "Image",
            "Predicted Class",
            "Confidence",
            "Benign Probability",
            "Malignant Probability",
        ],
        datatype=["str", "str", "str", "str", "str"],
        interactive=False,
        label="Batch Prediction Results",
    )

    batch_failures_output = gr.Markdown()

    batch_predict_button.click(
        fn=predict_batch_images,
        inputs=[
            multiple_images_input,
            folder_images_input,
        ],
        outputs=[
            batch_summary_output,
            batch_results_output,
            batch_failures_output,
        ],
    )


# ==========================================================
# APPLICATION ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    demo.launch(
        server_name=FRONTEND_HOST,
        server_port=FRONTEND_PORT,
    )
