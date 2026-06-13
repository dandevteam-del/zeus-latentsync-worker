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
    TMPDIR=/app/tmp

# Force all model caches onto /app (the container disk) instead of /root/.cache,
# so the runtime VAE/whisper downloads land on the big disk, not a small mount.
RUN mkdir -p /app/hfcache/hub /app/torchcache /app/tmp

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

# Weights are NOT baked at build (keeps the image small + the build reliable).
# handler.py downloads them on first cold start, cached on the worker.

ENV PYTHONPATH="/app/LatentSync:${PYTHONPATH}"
COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
