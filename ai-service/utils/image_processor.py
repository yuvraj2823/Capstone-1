"""
Image preprocessing utilities.
Decodes base64 image, resizes and normalizes for DenseNet-121.
"""
import base64
import logging
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalization replaced with XRV [-1024, 1024] scaling
TARGET_SIZE = (224, 224)

class XRayNormalize(object):
    def __call__(self, img_tensor):
        return img_tensor * 2048.0 - 1024.0

def resize_with_padding(img):
    # Maintain aspect ratio and pad with black
    img.thumbnail(TARGET_SIZE, Image.LANCZOS)
    new_img = Image.new("L", TARGET_SIZE, (0))
    new_img.paste(img, ((TARGET_SIZE[0] - img.size[0]) // 2,
                        (TARGET_SIZE[1] - img.size[1]) // 2))
    return new_img

_transform = transforms.Compose([
    transforms.Lambda(resize_with_padding),
    transforms.ToTensor(),
    XRayNormalize(),
])


def preprocess_image(
    image_b64: str, mime_type: str = "image/jpeg"
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Decodes a base64 image string, converts to RGB, and prepares for inference.

    Args:
        image_b64: Base64-encoded image (no data URI prefix needed, but tolerated).
        mime_type:  MIME type hint (used for logging only).

    Returns:
        image_tensor: Normalized float32 tensor of shape (1, 3, 224, 224).
        original_np:  Uint8 numpy array (H, W, 3) for Grad-CAM overlay.
    """
    # Strip data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc

    try:
        pil_image = Image.open(BytesIO(raw)).convert("L")  # Grayscale for XRV
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    logger.debug(f"Loaded image: {pil_image.size}, mode={pil_image.mode}, type={mime_type}")

    # Keep a resized uint8 copy (converted to RGB) for overlay generation
    original_resized = pil_image.resize(TARGET_SIZE, Image.BILINEAR).convert("RGB")
    original_np = np.array(original_resized, dtype=np.uint8)

    # Apply transforms
    tensor = _transform(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

    return tensor, original_np
