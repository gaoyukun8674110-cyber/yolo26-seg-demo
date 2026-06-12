# ---- YOLO26 GPU inference API ----
# Build a CUDA-enabled image that runs the FastAPI service with the REAL
# ultralytics model. The trained weight is NOT baked into the image; it is
# mounted at runtime and pointed to via YOLO26_MODEL_PATH (see docker-compose).
#
# Base: PyTorch + CUDA 12.4 runtime (matches the locally trained torch 2.5.1+cu124).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# OpenCV runtime needs libgl/libglib; install minimal system deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching. ultralytics is added on top of
# the API requirements because the runtime image needs real inference.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt \
    && pip install --no-cache-dir --no-deps ultralytics==8.4.42 ultralytics-thop \
    && pip install --no-cache-dir \
        "numpy>=2.2,<2.3.0" "pyyaml" "matplotlib" "pandas" "polars" "psutil" "py-cpuinfo" "requests" "scipy" "tqdm"

# Copy application source (weights/data are mounted at runtime, not copied).
COPY api /app/api

# artifacts/generated is written at runtime; ensure the dir exists.
RUN mkdir -p /app/artifacts/generated

EXPOSE 8000

# YOLO26_MODEL_PATH is injected by docker-compose / docker run.
CMD ["python", "-m", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
