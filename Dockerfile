# Zeus LatentSync worker — RunPod serverless GPU image (built by RunPod from this repo).
# Recipe matches a known-working LatentSync serverless build: CUDA 12.1, py3.10,
# torch 2.5.1, then LatentSync's own requirements, then weights baked at build.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LATENTSYNC_DIR=/app/LatentSync \
    LATENTSYNC_UNET=/app/LatentSync/checkpoints/latentsync_unet.pt \
    LATENTSYNC_HF_REPO=ByteDance/LatentSync-1.6 \
    HF_HOME=/app/hfcache \
    HUGGINGFACE_HUB_CACHE=/app/hfcache/hub \
    TORCH_HOME=/app/torchcache \
    TMPDIR=/app/tmp \
    HOME=/app \
    INSIGHTFACE_HOME=/app/.insightface

# /root is a small mount; /app holds the big container disk (the 5GB unet
# download lands there fine). Redirect EVERY home-relative cache (HF default,
# insightface face-detector, torch, matplotlib...) to /app by setting HOME=/app,
# so no runtime model download ever touches the small /root mount.
RUN mkdir -p /app/hfcache/hub /app/torchcache /app/tmp /app/.insightface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev build-essential \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch first (cu121), matching the LatentSync-tested versions.
RUN pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121

# LatentSync (ByteDance, OpenRAIL++ — commercial use OK with attribution).
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git /app/LatentSync
RUN pip install -r /app/LatentSync/requirements.txt && \
    pip install runpod huggingface_hub tensorflow-cpu

# The big LatentSync unet (5GB) is NOT baked (downloaded at cold start). But the
# small VAE the inference script loads via diffusers (~335MB) IS baked here, into
# the same HF cache the runtime uses — guaranteed present, no runtime download,
# no cache-path race. This was the recurring inference failure.
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('stabilityai/sd-vae-ft-mse', \
allow_patterns=['*.safetensors','*.bin','*.json'])"

ENV PYTHONPATH="/app/LatentSync:${PYTHONPATH}"
COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
