
import torch

from huggingface_hub import hf_hub_download

from models import FibonacciNet

from utils.config import (
    REPO_ID,
    MODEL_FILENAME,
    INPUT_CHANNELS,
    NUM_CLASSES,
    DEVICE
)


def load_model():

    # Download checkpoint from Hugging Face
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )

    # Initialize architecture
    model = FibonacciNet(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES
    )

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Extract model parameters
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model, DEVICE
