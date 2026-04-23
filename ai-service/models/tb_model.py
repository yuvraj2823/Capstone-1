"""
TB Classification Model using Transfer Learning.
Loads a fine-tuned DenseNet-121 model for binary TB classification.
If no model file exists, initialises with pretrained ImageNet weights
so the service remains functional for demonstration purposes.
"""
import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from torchvision import models

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "tb_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TBClassifier(nn.Module):
    """DenseNet-121 backbone with a binary classification head."""

    def __init__(self):
        super().__init__()
        base = xrv.models.DenseNet(weights=None)

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base.classifier.in_features, 1),
        )
        # Hook storage for Grad-CAM
        self._feature_maps: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def save_gradient(self, grad: torch.Tensor):
        self._gradients = grad.detach()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fmaps = F.relu(self.features(x), inplace=True)

        # Register backward hook to capture gradients
        if fmaps.requires_grad:
            fmaps.register_hook(self.save_gradient)

        pooled = self.avgpool(fmaps)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        return torch.sigmoid(out).squeeze(1), fmaps


class TBModel:
    """Wraps TBClassifier with load/predict interface."""

    def __init__(self):
        self.device = DEVICE
        self.net: TBClassifier | None = None

    def load(self):
        self.net = TBClassifier().to(self.device)

        if os.path.isfile(MODEL_PATH):
            logger.info(f"CRITICAL: Found weights file at {MODEL_PATH}. Loading...")
            print(f"\n[AI-SERVICE] >>> LOADING MODEL WEIGHTS FROM: {MODEL_PATH} <<<\n")
            state = torch.load(MODEL_PATH, map_location=self.device)
            
            if isinstance(state, dict) and "model_state_dict" in state:
                self.net.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "state_dict" in state:
                self.net.load_state_dict(state["state_dict"])
            else:
                self.net.load_state_dict(state)
            
            print("[AI-SERVICE] ✅ WEIGHTS LOADED SUCCESSFULLY INTO MODEL\n")
        else:
            print(f"\n[AI-SERVICE] ❌ ERROR: WEIGHTS NOT FOUND AT {MODEL_PATH}")
            print("[AI-SERVICE] ❌ THE MODEL IS RUNNING ON RANDOM BRAINS. STOPPING SERVICE.\n")
            raise FileNotFoundError(f"Trained model weights missing at {MODEL_PATH}")

        self.net.eval()
        logger.info(f"TBClassifier ready on {self.device}.")

    def predict_with_gradients(
        self, image_tensor: torch.Tensor
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Runs forward pass and captures Grad-CAM intermediate values.

        Hooks into `denseblock3` (14×14 feature maps) for much better
        spatial localization than the final 7×7 output layer.

        Returns (prediction_score, feature_maps, gradients).
        """
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        image_tensor = image_tensor.to(self.device)

        # Storage for hook outputs
        _fmaps: dict = {}
        _grads: dict = {}

        def _fwd_hook(module, input, output):
            _fmaps['value'] = output

        def _bwd_hook(module, grad_input, grad_output):
            _grads['value'] = grad_output[0].detach()

        # Hook denseblock3 → 14×14 spatial resolution (much better than 7×7)
        target_layer = self.net.features.denseblock3
        fwd_handle = target_layer.register_forward_hook(_fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

        try:
            with torch.enable_grad():
                image_tensor = image_tensor.requires_grad_(True)
                self.net.zero_grad()

                # Full forward pass through the model
                feat = F.relu(self.net.features(image_tensor), inplace=False)
                pooled = self.net.avgpool(feat)
                flat = torch.flatten(pooled, 1)
                logit = self.net.classifier(flat)
                score = torch.sigmoid(logit).squeeze()

                # Backprop INSIDE enable_grad context to capture real gradients
                # Use score (post-sigmoid) for correct Grad-CAM w.r.t. predicted class
                score.backward()

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

        feature_maps = _fmaps.get('value', torch.zeros(1, 1, 14, 14)).detach()
        gradients   = _grads.get('value', torch.zeros_like(feature_maps))

        return float(score.item()), feature_maps, gradients

