
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from backend.routes import router

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.exception_handlers import (
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)

from models.model_loader import load_model

from services.inference_service import InferenceService
from services.batch_inference_service import BatchInferenceService

from utils.config import DEVICE, CLASSIFICATION_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Thyroid Detection API...")

    # Load trained model from Hugging Face.
    print("Loading FibonacciNet model...")

    model, DEVICE = load_model()

    print("FibonacciNet loaded successfully.")

    # Initialize single-image inference service.
    inference_service = InferenceService(
        model=model,
        device=DEVICE,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    # Initialize batch inference service.
    batch_inference_service = BatchInferenceService(
        inference_service=inference_service
    )

    # Store services in application state.
    app.state.model = model
    app.state.inference_service = inference_service
    app.state.batch_inference_service = batch_inference_service

    print("Inference services initialized.")
    print("Thyroid Detection API is ready.")

    yield

    # Application shutdown.
    print("Shutting down Thyroid Detection API...")

    app.state.model = None
    app.state.inference_service = None
    app.state.batch_inference_service = None

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Use OpenAPI 3.0.3 for Swagger UI file upload compatibility.
    openapi_schema["openapi"] = "3.0.3"

    schemas = openapi_schema.get("components", {}).get("schemas", {})

    for schema in schemas.values():

        properties = schema.get("properties", {})

        for property_schema in properties.values():

            # Handle single-file uploads.
            if property_schema.get("contentMediaType") == "application/octet-stream":

                property_schema.pop("contentMediaType", None)
                property_schema["format"] = "binary"

            # Handle arrays of uploaded files.
            items = property_schema.get("items", {})

            if items.get("contentMediaType") == "application/octet-stream":

                items.pop("contentMediaType", None)
                items["format"] = "binary"

    app.openapi_schema = openapi_schema

    return app.openapi_schema



app = FastAPI(
    title="Thyroid Nodule Detection API",
    description=(
        "API for thyroid ultrasound image classification "
        "and Grad-CAM explainability using FibonacciNet."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_exception_handler(
    StarletteHTTPException,
    http_exception_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)

app.add_exception_handler(
    Exception,
    general_exception_handler,
)

app.include_router(
    router,
    prefix="/api/v1",
)

app.openapi = custom_openapi