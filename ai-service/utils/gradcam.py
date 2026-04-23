"""
Grad-CAM heatmap generation.
Produces a coloured heatmap and an overlay on the original image,
both returned as base64-encoded JPEG strings.
"""
import base64
import logging
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def generate_gradcam(
    original_np: np.ndarray,
    feature_maps: torch.Tensor,
    gradients: torch.Tensor,
    alpha: float = 0.45,
    percentile_threshold: float = 40.0,
) -> Tuple[str, str]:
    """
    Generates Grad-CAM heatmap and blended overlay image.

    Args:
        original_np:          uint8 numpy array (H, W, 3) – original resized image.
        feature_maps:         Tensor (1, C, H', W') – last conv feature maps.
        gradients:            Tensor (1, C, H', W') – gradients w.r.t. the output of denseblock3
        alpha:                Blend factor for overlay (0 = only original, 1 = only heatmap).
        percentile_threshold: Activations below this percentile are zeroed out to suppress
                              background noise (hands, edges, etc.).

    Returns:
        heatmap_b64:  Base64-encoded JPEG of the Grad-CAM heatmap.
        overlay_b64:  Base64-encoded JPEG of heatmap blended with original.
    """
    # ─── Compute Grad-CAM weights (standard algorithm) ────────────────────────
    # Global average pool gradients → one scalar weight per channel
    weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

    # Weighted combination of feature maps — single ReLU on the result only.
    # Do NOT ReLU the weights first; that is a non-standard deviation that
    # throws away negatively-weighted channels and produces flat/uniform maps.
    cam = (weights * feature_maps).sum(dim=1, keepdim=False)  # (1, H', W')
    cam = torch.relu(cam)                                       # keep only positive activations
    cam = cam.squeeze().cpu().numpy()                           # (H', W')  or scalar if H'=W'=1

    # Guard: if feature map collapsed to a scalar (shouldn't happen with denseblock3)
    if cam.ndim == 0:
        cam = np.full((7, 7), float(cam))

    # ─── Suppress background noise via percentile thresholding ────────────────
    if cam.max() > 0:
        # Use a very low threshold (10th percentile) to avoid masking real signals
        threshold_val = np.percentile(cam, percentile_threshold if percentile_threshold else 10.0)
        cam = np.where(cam >= threshold_val, cam, 0.0)
        if cam.max() > 0:
            cam = cam / cam.max()
    else:
        # Gradients were zero or all-negative — return blank overlay
        logger.warning("Grad-CAM produced a zero activation map. Returning blank heatmap.")
        cam = np.zeros_like(cam, dtype=np.float32)

    # ─── Soften edges (as requested to not be "too smooth") ──────────────────
    cam = cam.astype(np.float32)
    # sigma=0.3 provides a very subtle softening to hide grid artifacts
    # without making the map "fuzzy" or "too smooth".
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=0.3, sigmaY=0.3)
    
    # Re-normalise (safeguard)
    if cam.max() > 0:
        cam = cam / cam.max()

    # ─── Resize to match original image ───────────────────────────────────────
    h, w = original_np.shape[:2]
    # Cubic is standard for Grad-CAM overlays
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8 = np.uint8(255 * cam_resized)

    # ─── Apply colour map (TURBO: better perceptual clarity than JET) ─────────
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ─── Overlay: only blend where heatmap is strong (mask-based) ────────────
    # Create a strength mask so low-activation background stays clear
    strength_mask = cam_resized[..., np.newaxis]  # (H, W, 1)
    blended = (
        (1 - alpha * strength_mask) * original_np.astype(np.float32)
        + (alpha * strength_mask) * heatmap_rgb.astype(np.float32)
    )
    overlay_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    # ─── Encode to base64 JPEG ────────────────────────────────────────────────
    heatmap_b64 = _numpy_to_b64_jpeg(heatmap_rgb)
    overlay_b64 = _numpy_to_b64_jpeg(overlay_rgb)

    return heatmap_b64, overlay_b64



def _numpy_to_b64_jpeg(arr: np.ndarray, quality: int = 90) -> str:
    """Converts an (H, W, 3) uint8 numpy array to a base64-encoded JPEG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
