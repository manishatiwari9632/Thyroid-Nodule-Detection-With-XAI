
from pydantic import BaseModel


# ==========================================================
# HEALTH RESPONSE SCHEMAS
# ==========================================================

class HealthData(BaseModel):
    model_loaded: bool
    inference_service_loaded: bool
    batch_inference_service_loaded: bool


class HealthResponse(BaseModel):
    success: bool
    message: str
    data: HealthData


# ==========================================================
# PREDICTION RESPONSE SCHEMAS
# ==========================================================

class PredictionData(BaseModel):
    predicted_label: int
    predicted_class: str
    benign_probability: float
    malignant_probability: float
    confidence: float
    threshold: float


# ==========================================================
# EXPLANATION RESPONSE SCHEMAS
# ==========================================================

class ExplanationData(BaseModel):
    explained_class: int
    explained_label: str
    heatmap: str
    overlay: str


# ==========================================================
# SINGLE IMAGE RESPONSE SCHEMAS
# ==========================================================

class SinglePredictionData(BaseModel):
    filename: str
    prediction: PredictionData
    explanation: ExplanationData


class SinglePredictionResponse(BaseModel):
    success: bool
    message: str
    data: SinglePredictionData


# ==========================================================
# BATCH RESPONSE SCHEMAS
# ==========================================================

class BatchSummary(BaseModel):
    total_images: int
    successful_count: int
    failed_count: int


class FailedImage(BaseModel):
    path: str
    reason: str


class BatchPredictionData(BaseModel):
    summary: BatchSummary
    results: list[SinglePredictionData]
    failed_images: list[FailedImage]


class BatchPredictionResponse(BaseModel):
    success: bool
    message: str
    data: BatchPredictionData
