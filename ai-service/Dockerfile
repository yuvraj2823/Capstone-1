# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Force total rebuild with no cache
# Increment this number to trigger a 100% fresh build
ARG CACHE_BUST=20260424

# Install system dependencies
# libgl1        – required by opencv-python-headless for cv2 color operations (Grad-CAM)
# libglib2.0-0  – required by opencv for threading primitives
# libsm6/libxext6/libxrender-dev – X11 stubs needed by OpenCV even in headless mode
# libgomp1      – OpenMP runtime used by PyTorch CPU kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Ensure Python output is not buffered (visible in docker logs immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Cache-bust: Change this version string to force a code-only rebuild ---
LABEL build-version="2026-04-24-FINAL"
COPY . .

# Create models directory
RUN mkdir -p /app/models

EXPOSE 8000

# Increased start_period to 60s — DenseNet weight loading takes ~30-40s on CPU
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
