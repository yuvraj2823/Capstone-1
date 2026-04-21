"""
Pydantic schemas for the AI service.
"""
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded chest X-ray image (JPEG/PNG/WebP).",
        min_length=100,
    )
    mime_type: str = Field(
        default="image/jpeg",
        description="Image MIME type.",
    )

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
        if v not in allowed:
            raise ValueError(f"mime_type must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    prediction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="TB probability score (0–1). >0.5 indicates TB positive.",
    )
    heatmap: str = Field(
        ...,
        description="Base64-encoded JPEG Grad-CAM heatmap.",
    )
    overlay: str = Field(
        ...,
        description="Base64-encoded JPEG Grad-CAM overlay on original image.",
    )
    label: str = Field(
        default="",
        description="Human-readable label: 'TB Positive' or 'TB Negative'.",
    )
    confidence: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Confidence percentage.",
    )

    def model_post_init(self, __context) -> None:
        self.label = "TB Positive" if self.prediction > 0.5 else "TB Negative"
        self.confidence = round(max(self.prediction, 1 - self.prediction) * 100)
