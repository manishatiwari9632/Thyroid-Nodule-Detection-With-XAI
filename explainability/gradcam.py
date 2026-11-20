
import numpy as np
import torch
import torch.nn.functional as F

from utils.config import DEVICE


class GradCAM:
    """
    Grad-CAM implementation for FibonacciNet.

    Uses model.block7 as the default target layer and preserves
    the Grad-CAM calculation from the original notebook.
    """

    def __init__(
        self,
        model,
        target_layer=None,
        device=DEVICE
    ):
        self.model = model
        self.device = torch.device(device)

        # Use Block 7 unless another target layer is explicitly provided.
        self.target_layer = (
            target_layer
            if target_layer is not None
            else self.model.block7
        )

        self.model.to(self.device)

    def generate_heatmap(
        self,
        img_tensor: torch.Tensor,
        pred_index=None
    ) -> np.ndarray:
        """
        Generate a normalized Grad-CAM heatmap.

        Args:
            img_tensor:
                Input tensor with shape [1, C, H, W].
            pred_index:
                For binary classification:
                    0 -> Benign class explanation
                    1 -> Malignant class explanation
                    None -> Positive class (class 1)

                For multiclass classification:
                    None -> Predicted class
                    Integer -> Selected class

        Returns:
            Normalized NumPy heatmap with shape [H, W].
        """

        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Expected img_tensor to be a PyTorch tensor")

        if img_tensor.ndim != 4:
            raise ValueError(
                "Expected input tensor with shape [1, C, H, W]"
            )

        if img_tensor.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently supports one image at a time"
            )

        activations = {}

        def forward_hook(module, inputs, output):
            activations["features"] = output

        hook_handle = self.target_layer.register_forward_hook(
            forward_hook
        )

        was_training = self.model.training
        self.model.eval()

        try:
            img_tensor = img_tensor.to(self.device)

            # Clear any previously accumulated parameter gradients.
            self.model.zero_grad(set_to_none=True)

            # Grad-CAM requires autograd, even if called from a
            # surrounding torch.no_grad() context.
            with torch.enable_grad():

                output = self.model(img_tensor)

                if "features" not in activations:
                    raise RuntimeError(
                        "Target layer did not produce feature maps"
                    )

                feature_map = activations["features"]

                if output.ndim != 2:
                    raise RuntimeError(
                        "Expected model output with shape [batch, classes]"
                    )

                if output.shape[1] == 1:

                    # Single-logit binary classification.
                    if pred_index not in (None, 0, 1):
                        raise ValueError(
                            "For binary classification, pred_index "
                            "must be 0, 1, or None"
                        )

                    if pred_index == 0:
                        class_score = -output[:, 0]
                    else:
                        class_score = output[:, 0]

                else:

                    # Multiclass classification support.
                    if pred_index is None:
                        pred_index = output.argmax(dim=1).item()

                    if not 0 <= pred_index < output.shape[1]:
                        raise ValueError(
                            "pred_index is outside the model's "
                            "output class range"
                        )

                    class_score = output[:, pred_index]

                gradients = torch.autograd.grad(
                    outputs=class_score.sum(),
                    inputs=feature_map,
                    retain_graph=False,
                    create_graph=False
                )[0]

                # Global average pooling of gradients.
                pooled_grads = torch.mean(
                    gradients,
                    dim=(0, 2, 3)
                )

                # Extract feature maps for the input image.
                feature_maps = feature_map.detach()[0]

                # Weight feature maps using pooled gradients.
                heatmap = torch.sum(
                    feature_maps * pooled_grads[:, None, None],
                    dim=0
                )

                # Apply ReLU.
                heatmap = F.relu(heatmap)

                # Normalize heatmap to [0, 1].
                max_val = torch.max(heatmap)

                if max_val > 0:
                    heatmap = heatmap / max_val

                return heatmap.detach().cpu().numpy().astype(
                    np.float32
                )

        finally:
            # Always remove the forward hook.
            hook_handle.remove()

            # Restore the model's original training/evaluation state.
            self.model.train(was_training)
