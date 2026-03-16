FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    MODEL_ROOT=/opt/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    tini \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install huggingface_hub openvino

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "-s", "--"]
CMD ["sleep", "infinity"]
