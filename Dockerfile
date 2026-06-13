# Zeus LatentSync worker — RunPod serverless GPU image (built by RunPod from this repo).
# Small image (code + deps only). ALL model storage + caches + temp live on the
# attached NETWORK VOLUME (/runpod-volume), NOT the tiny container disk — this is
# the definitive fix for the recurring "No space left on device".
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LATENTSYNC_DIR=/app/LatentSync \
    LATENTSYNC_HF_REPO=ByteDance/LatentSync-1.6 \
    LATENTSYNC_UNET=/runpod-volume/ls-ckpt/latentsync_unet.pt \
    HOME=/runpod-volume \
    HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf/hub \
    TORCH_HOME=/runpod-volume/torch \
    INSIGHTFACE_HOME=/runpod-volume/insightface \
    TMPDIR=/runpod-volume/tmp

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev build-essential \
        git ffmpeg libgl1 libglib2.0-0 ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121

# LatentSync (ByteDance, OpenRAIL++ — commercial use OK with attribution).
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git /app/LatentSync
RUN pip install -r /app/LatentSync/requirements.txt && \
    pip install runpod huggingface_hub tensorflow-cpu

# Models are NOT baked — they download to the network volume on first cold start
# (persistent, large, written once). handler.py sets up the volume dirs + the
# checkpoints symlink so LatentSync's relative paths resolve onto the volume.
ENV PYTHONPATH="/app/LatentSync:${PYTHONPATH}"
COPY handler.py /app/handler.py
CMD ["python3", "-u", "/app/handler.py"]
