import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

"""
TB Detection AI Service
FastAPI application with model loaded once at startup.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.tb_model import TBModel
from schemas.prediction import PredictionRequest, PredictionResponse
from utils.image_processor import preprocess_image
from utils.gradcam import generate_gradcam

# Version: 1.0.1 (Force Reload)
print("\n--- AI SERVICE BOOTING UP ---")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-service")

# ─── Global model instance ────────────────────────────────────────────────────
model: TBModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release at shutdown."""
    global model
    logger.info("Loading TB detection model...")
    try:
        model = TBModel()
        model.load()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        # Allow app to start even without model (returns 503 on predict)
        model = None
    yield
    logger.info("Shutting down AI service.")
    model = None


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TB Detection AI Service",
    description="Chest X-ray TB detection with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Error handler ────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["monitoring"])
async def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False},
        )
    return {"status": "healthy", "model_loaded": True}


# ─── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest):
    """
    Accepts a base64-encoded chest X-ray image.
    Returns:
      - prediction: float (0–1, >0.5 = TB positive)
      - heatmap:   base64 JPEG Grad-CAM heatmap
      - overlay:   base64 JPEG Grad-CAM overlay on original image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check the service logs.",
        )

    try:
        # Preprocess
        image_tensor, original_np = preprocess_image(request.image, request.mime_type)

        # Run inference
        prediction_score, feature_maps, gradients = model.predict_with_gradients(image_tensor)

        # Generate Grad-CAM
        heatmap_b64, overlay_b64 = generate_gradcam(
            original_np, feature_maps, gradients
        )

        logger.info(f"Prediction complete. Score: {prediction_score:.4f}")

        return PredictionResponse(
            prediction=round(float(prediction_score), 6),
            heatmap=heatmap_b64,
            overlay=overlay_b64,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")
