
import os


# ==========================================================
# BACKEND API CONFIGURATION
# ==========================================================

BACKEND_BASE_URL = os.getenv(
    "BACKEND_BASE_URL",
    "http://127.0.0.1:8000",
)

API_V1_PREFIX = "/api/v1"


# ==========================================================
# API ENDPOINTS
# ==========================================================

HEALTH_ENDPOINT = f"{API_V1_PREFIX}/health"

PREDICT_ENDPOINT = f"{API_V1_PREFIX}/predict"

BATCH_PREDICT_ENDPOINT = f"{API_V1_PREFIX}/predict/batch"


# ==========================================================
# FRONTEND CONFIGURATION
# ==========================================================

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 7860

REQUEST_TIMEOUT = 120.0
