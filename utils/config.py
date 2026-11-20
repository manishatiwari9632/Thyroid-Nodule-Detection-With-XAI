import torch

# Hugging Face configuration
REPO_ID = "Ambica-Test-Dev/Thyroid-detection-Fibonacci-Net"
MODEL_FILENAME = "best_fibonacci_net.pth"

# Model configuration
INPUT_CHANNELS = 3
NUM_CLASSES = 1

# Image preprocessing
IMAGE_SIZE = 224

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# Classification
CLASSIFICATION_THRESHOLD = 0.50

CLASS_MAPPING = {
    0: "Benign",
    1: "Malignant"
}

# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_UPLOAD_SIZE_MB = 10
MAX_BATCH_SIZE_MB = 50

MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_BATCH_SIZE_BYTES = MAX_BATCH_SIZE_MB * 1024 * 1024