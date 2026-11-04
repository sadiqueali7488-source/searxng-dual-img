# ---------- 1. Frontend build ----------
FROM node:22-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- 2. Backend (FastAPI + PP-OCRv6 + OpenCLIP, CPU) ----------
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    HF_HOME=/models/huggingface \
    PADDLE_PDX_CACHE_HOME=/models/paddlex

# OpenCV / Paddle runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
# CPU-only PyTorch first (the default wheel would pull ~2 GB of CUDA libraries)
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./
COPY --from=frontend /app/frontend/dist /app/frontend/dist
RUN rm -rf /app/backend/cache /app/backend/__pycache__

# Results cache, job store and downloaded models live on volumes (see docker-compose.yml)
VOLUME ["/app/backend/cache", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fs http://localhost:8000/api/jobs > /dev/null || exit 1
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
