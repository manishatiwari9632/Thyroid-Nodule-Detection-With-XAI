
from pathlib import Path

import httpx

from frontend.config import (
    BACKEND_BASE_URL,
    HEALTH_ENDPOINT,
    PREDICT_ENDPOINT,
    BATCH_PREDICT_ENDPOINT,
    REQUEST_TIMEOUT,
)


class APIClientError(Exception):
    """Raised when communication with the backend API fails."""


class ThyroidAPIClient:
    """Client for communicating with the Thyroid Detection API."""

    def __init__(self):
        self.base_url = BACKEND_BASE_URL.rstrip("/")
        self.timeout = REQUEST_TIMEOUT

    def _handle_response(self, response: httpx.Response) -> dict:
        """Validate and return the backend JSON response."""

        try:
            response_data = response.json()
        except ValueError as exc:
            raise APIClientError(
                "Backend returned an invalid response."
            ) from exc

        if not response.is_success:
            message = response_data.get(
                "message",
                "Backend request failed.",
            )

            detail = response_data.get("detail")

            if detail:
                raise APIClientError(f"{message} Details: {detail}")

            raise APIClientError(message)

        return response_data

    def health_check(self) -> dict:
        """Check whether the backend is available."""

        try:
            response = httpx.get(
                f"{self.base_url}{HEALTH_ENDPOINT}",
                timeout=self.timeout,
            )

            return self._handle_response(response)

        except httpx.RequestError as exc:
            raise APIClientError(
                "Unable to connect to the backend API."
            ) from exc

    def predict_image(self, image_path: str) -> dict:
        """Submit a single image for prediction."""

        if not image_path:
            raise APIClientError("Please select an image.")

        file_path = Path(image_path)

        if not file_path.is_file():
            raise APIClientError("Selected image does not exist.")

        try:
            with file_path.open("rb") as image_file:

                files = {
                    "file": (
                        file_path.name,
                        image_file,
                    )
                }

                response = httpx.post(
                    f"{self.base_url}{PREDICT_ENDPOINT}",
                    files=files,
                    timeout=self.timeout,
                )

            return self._handle_response(response)

        except httpx.RequestError as exc:
            raise APIClientError(
                "Unable to communicate with the backend API."
            ) from exc

        except OSError as exc:
            raise APIClientError(
                "Unable to read the selected image."
            ) from exc

    def predict_batch(self, image_paths: list[str]) -> dict:
        """Submit multiple images for batch prediction."""

        if not image_paths:
            raise APIClientError("Please select at least one image.")

        opened_files = []

        try:
            files = []

            for image_path in image_paths:

                file_path = Path(image_path)

                if not file_path.is_file():
                    continue

                image_file = file_path.open("rb")
                opened_files.append(image_file)

                files.append(
                    (
                        "files",
                        (
                            file_path.name,
                            image_file,
                        ),
                    )
                )

            if not files:
                raise APIClientError(
                    "No valid image files were selected."
                )

            response = httpx.post(
                f"{self.base_url}{BATCH_PREDICT_ENDPOINT}",
                files=files,
                timeout=self.timeout,
            )

            return self._handle_response(response)

        except httpx.RequestError as exc:
            raise APIClientError(
                "Unable to communicate with the backend API."
            ) from exc

        except OSError as exc:
            raise APIClientError(
                "Unable to read the selected images."
            ) from exc

        finally:
            for image_file in opened_files:
                image_file.close()
