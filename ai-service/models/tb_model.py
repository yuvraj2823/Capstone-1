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
        Returns (prediction_score, feature_maps, gradients).
        """
        if self.net is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        _fmaps, _grads = {}, {}
        def _fwd_hook(mod, inp, out): _fmaps['value'] = out
        def _bwd_hook(mod, grad_in, grad_out): _grads['value'] = grad_out[0]

        # Hook denseblock3 — 14x14 spatial resolution provides much finer
        # localization of diagnostic features than the final 7x7 map.
        target_layer = self.net.features.denseblock3
        fwd_handle = target_layer.register_forward_hook(_fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

        try:
            with torch.enable_grad():
                x = image_tensor.to(self.device).detach().requires_grad_(True)
                self.net.zero_grad()
                prob, logit, _ = self.net(x)
                
                # Backprop on logit to avoid sigmoid saturation
                logit.backward()
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

        # denseblock3 output is 14x14
        feature_maps = _fmaps.get('value', torch.zeros(1, 1, 14, 14)).detach()
        gradients   = _grads.get('value', torch.zeros_like(feature_maps))

        # Increase threshold for TB detection (only flag if very sure)
        # We can also compare against the base XRV model here if needed
        return float(prob.item()), feature_maps, gradients
