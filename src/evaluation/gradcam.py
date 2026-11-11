"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Module
Generates visual explanations for ResNet-18 jersey crop feature representations.
Directly aligns with Computer Vision Syllabus Unit VI (Explainable AI & Saliency Maps).
"""

from typing import Tuple, Optional
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GradCAM:
    """
    Computes Grad-CAM saliency heatmaps for CNN feature layers (ResNet-18 layer4).
    """

    def __init__(self, model: Optional[nn.Module] = None, target_layer: Optional[nn.Module] = None, device: Optional[str] = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        if model is None:
            # Load default ResNet-18
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        else:
            self.model = model.to(self.device)

        self.model.eval()

        # Target the final residual convolutional block of ResNet-18
        self.target_layer = target_layer or self.model.layer4[-1]

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self._register_hooks()

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        crop_bgr: np.ndarray,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM activation heatmap for a player crop.

        Args:
            crop_bgr: BGR crop image (e.g. 224x224).
            target_class_idx: Optional target class index. If None, uses top prediction.

        Returns:
            Tuple of (grayscale_heatmap [0..1], blended_overlay_bgr uint8).
        """
        h, w = crop_bgr.shape[:2]

        # Preprocess input tensor
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        norm_tensor = (tensor - self.mean) / self.std

        # Forward pass
        output = self.model(norm_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(output, dim=1).item())

        # Backward pass for target class score
        self.model.zero_grad()
        score = output[0, target_class_idx]
        score.backward()

        # Pool gradients across spatial dimensions (global average pooling of gradients)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activation channels by corresponding pooled gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Sum weighted activations and apply ReLU
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize heatmap to [0, 1]
        heatmap_np = heatmap.cpu().detach().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np = heatmap_np / max_val
        else:
            heatmap_np = np.zeros_like(heatmap_np)

        # Resize heatmap to match original crop dimensions
        resized_heatmap = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Colorize heatmap with JET colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)

        # Alpha blend overlay
        overlay = cv2.addWeighted(crop_bgr, 0.6, heatmap_colored, 0.4, 0)

        return resized_heatmap, overlay

    def save_explanation(
        self,
        crop_bgr: np.ndarray,
        output_path: str,
        title: str = "ResNet-18 Grad-CAM Explainability"
    ) -> str:
        """
        Generate and save a side-by-side comparison (Original | Heatmap | Overlay).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _, overlay = self.generate_heatmap(crop_bgr)

        h, w = crop_bgr.shape[:2]
        # Side-by-side canvas: Original Crop | Grad-CAM Overlay
        comparison = np.zeros((h + 40, w * 2 + 20, 3), dtype=np.uint8)

        # Top banner
        cv2.putText(comparison, title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)

        # Place images
        comparison[40:40 + h, 5:5 + w] = crop_bgr
        comparison[40:40 + h, w + 15:w * 2 + 15] = overlay

        # Sublabels
        cv2.putText(comparison, "Original Crop", (20, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(comparison, "Grad-CAM Saliency", (w + 30, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, comparison)
        return output_path
