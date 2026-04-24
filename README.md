# TB-Detect: AI-Powered Tuberculosis Screening
### Advanced Chest X-Ray Diagnostics with Explainable AI (Grad-CAM) & Dynamic Reporting

---

## Technology Stack
- **Frontend**: React 18, Daikon (DICOM), React-App-Rewired, Axios.
- **Backend**: Node.js/Express, Mongoose, Opossum (Circuit Breaker), Sharp, PDFKit.
- **AI Service**: FastAPI, PyTorch, TorchXRayVision (DenseNet-121), OpenCV.
- **DevOps**: Docker, Jenkins CI/CD, MongoDB 7.0, Pytest/Jest.

## Architecture
TB-Detect is built as a robust microservices-based system, containerized for scale and consistency.

```
┌─────────────────────────────────────────────────────────────┐
│                   tb-net (Docker Network)                   │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │          │    │              │    │                 │    │
│  │ frontend │──▶ │   backend    │──▶ │   ai-service    │    │
│  │ React    │    │  Node+Express│    │  FastAPI+PyTorch│    │
│  │ :3000    │    │  :5000       │    │  :8000          │    │
│  └──────────┘    └──────┬───────┘    └─────────────────┘    │
│                         │                                   │
│                   ┌─────▼──────┐                            │
│                   │   mongo    │                            │
│                   │  MongoDB   │                            │
│                   │  :27017    │                            │
│                   └────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## Key Features
- **AI Diagnostics**: High-accuracy TB detection using DenseNet-121 (Pre-trained on X-rays via TorchXRayVision).
- **Explainable AI (XAI)**: Optimized **Grad-CAM** pipeline with positive-influence gradient filtering to focus specifically on pathological indicators.
- **DICOM Support**: Native parsing and rendering of `.dcm` standard medical images using Daikon.
- **Advanced Radiological Viewer**: Interactive workstation-style image viewer with Brightness, Contrast, Zoom, Pan, Invert, and adjustable Heatmap Opacity sliders.
- **Auto-Patient ID**: Dynamic generation of unique Patient IDs (`P-XXXXXX`) for every diagnostic session.
- **Diagnostic Reports**: Instant PDF report generation with side-by-side original/heatmap/overlay grids.
- **History & Analytics**: Complete record-keeping and an interactive dashboard for trend analysis.
- **Data Privacy**: Automatic stripping of EXIF metadata and secure JWT-based authentication.
- **Reliability**: Integrated **Circuit Breaker** (Opossum) to ensure system stability during AI service timeouts.

## Environment Configuration
Before running the services, ensure you have a `.env` file in the root or specific service directories with the following:
```env
MONGO_USER=admin
MONGO_PASSWORD=secret
MONGO_DB=tbdetection
JWT_SECRET=your_secure_jwt_key
AI_SERVICE_URL=http://localhost:8000
```

## Quick Start (Local Development)

### 1. Prerequisites
- Node.js (v18+)
- Python (3.9+)
- MongoDB (running locally or via Docker)

### 2. Launching Services

**AI Service (Python/FastAPI):**
```bash
cd ai-service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --port 8000 --reload
```

**Backend (Node.js/Express):**
```bash
cd backend
npm install
npm run dev
```

**Frontend (React):**
```bash
cd frontend
npm install
npm start
```

## AI Training Pipeline
To retrain the model on your own dataset (e.g., TBX11K):
1.  Organize images into `ai-service/dataset/TBX11K/imgs/` under `health/`, `sick/`, and `tb/` folders.
2.  Run the training script:
    ```bash
    cd ai-service
    python train.py
    ```
    *The script includes Early Stopping, Learning Rate Scheduling, and F1-Score evaluation.*

## CI/CD with Jenkins
The included `Jenkinsfile` automates the entire pipeline:
1.  **Code Quality**: Lints and checks for both JS and Python.
2.  **Unit Tests**: Runs Jest (Backend) and Pytest (AI Service).
3.  **Integration**: Validates service communication.
4.  **Containerization**: Builds Docker images for all services.
5.  **Deployment**: Automated rollout via Docker Compose.

## Docker Deployment
```bash
docker-compose up --build -d
```

## API Reference
- **Auth**: `POST /api/auth/login`, `POST /api/auth/signup`
- **Diagnostics**: `POST /api/upload` (Analyze X-ray)
- **History**: `GET /api/history` (List records), `GET /api/history/report/:id` (Download PDF)
- **Analytics**: `GET /api/analytics/stats`

---
**Disclaimer**: This system is for **research and educational purposes only**. It is not a certified medical diagnostic tool.
