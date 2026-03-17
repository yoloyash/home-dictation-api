FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    MODEL_ROOT=/opt/models \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tini \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install av fastapi httpx huggingface_hub openvino pytest python-multipart uvicorn

WORKDIR /app

COPY . /app

ENTRYPOINT ["/usr/bin/tini", "-s", "--"]
CMD ["sh", "-c", "python scripts/bootstrap_model.py && exec uvicorn home_dictation_api.api:app --host 0.0.0.0 --port 8080"]
