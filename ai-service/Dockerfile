# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies
# libgl1        – required by opencv-python-headless for cv2 color operations
# libglib2.0-0  – required by opencv for threading primitives
# libsm6/libxext6/libxrender-dev – X11 stubs needed by OpenCV even in headless mode
# libgomp1      – OpenMP runtime used by PyTorch CPU kernels
# curl          – needed for the container healthcheck
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

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Cache-bust: Updated version to force a code-only rebuild ---
LABEL build-version="2026-04-24-v1"
COPY . .

# Create models directory to ensure model weights have a home
RUN mkdir -p /app/models

EXPOSE 8000

# Healthcheck to monitor model availability
# start_period is 60s because DenseNet weights take time to load on CPU
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
