# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0 \
    JOYOMNI_OPS_CUDA_ARCHS=120a \
    MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /opt/joyai

COPY deploy/requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt

# Build the repository's original FP8 CUDA operations.
COPY deploy/joyomni_ops /opt/joyai/deploy/joyomni_ops

RUN git clone https://github.com/NVIDIA/cutlass.git /tmp/cutlass \
    && git -C /tmp/cutlass checkout dcf215af \
    && JOYOMNI_OPS_CUTLASS_DIR=/tmp/cutlass \
       python3 -m pip install --no-build-isolation /opt/joyai/deploy/joyomni_ops \
    && rm -rf /tmp/cutlass /root/.cache/pip

COPY . /opt/joyai

RUN chmod +x /opt/joyai/deploy/run_server.sh \
    && mkdir -p /runpod-volume/joyai \
    && mkdir -p /tmp/joyomni-recordings

ENV JOYOMNI_DEVICE=cuda:0 \
    JOYOMNI_HOST=0.0.0.0 \
    JOYOMNI_PORT=8080 \
    JOYOMNI_CKPT_ROOT=/runpod-volume/joyai/checkpoints \
    JOYOMNI_CACHE_ROOT=/runpod-volume/joyai/cache/rtx-pro-6000-blackwell-torch291-cu128-oomfix-v2 \
    JOYOMNI_CACHE_READY_MARKER=/runpod-volume/joyai/cache/rtx-pro-6000-blackwell-torch291-cu128-oomfix-v2/ready.json \
    JOYOMNI_EXPECTED_CUDA_CAPABILITY=12.0 \
    JOYOMNI_PRELOAD=1 \
    JOYOMNI_WIDTH=840 \
    JOYOMNI_HEIGHT=480 \
    JOYOMNI_FPS=24 \
    JOYOMNI_NUM_INFERENCE_STEPS=2 \
    JOYOMNI_FP8_IMG=1 \
    JOYOMNI_FP8_TXT=1 \
    JOYOMNI_CUDA_GRAPH=1 \
    JOYOMNI_SAGE_ATTN=0 \
    JOYOMNI_TXT_PARALLEL=0 \
    JOYOMNI_FP8_FAST_ACCUM=0 \
    JOYOMNI_VAE_COMPILE=1 \
    JOYOMNI_VAE_COMPILE_STRICT=1 \
    JOYOMNI_LOAD_WARMUP_STRICT=1 \
    JOYOMNI_FULL_WARMUP_TIMEOUT_SECONDS=300 \
    JOYOMNI_WARMUP_BOTH_ORIENTATIONS=0 \
    JOYOMNI_WARMUP_REFERENCE_BUCKETS=1 \
    JOYOMNI_RECORD_DIR=/tmp/joyomni-recordings \
    JOYOMNI_RECORD_ENABLED=0 \
    JOYOMNI_ONLINE_GATE_ENABLED=0

EXPOSE 8080 8081

CMD ["python3", "/opt/joyai/runpod/start.py"]
