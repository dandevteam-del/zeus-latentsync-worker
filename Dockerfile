# Zeus LatentSync worker — RunPod serverless GPU image (built by RunPod from this repo).
# Bakes ByteDance LatentSync + weights at build time so cold starts are fast and
# no network volume is required.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LATENTSYNC_DIR=/app/LatentSync \
    LATENTSYNC_UNET=/app/LatentSync/checkpoints/latentsync_unet.pt \
    LATENTSYNC_CONFIG=/app/LatentSync/configs/unet/stage2.yaml \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# LatentSync (ByteDance, OpenRAIL++ — commercial use OK with attribution).
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git /app/LatentSync
RUN pip install --no-cache-dir -r /app/LatentSync/requirements.txt && \
    pip install --no-cache-dir runpod huggingface_hub hf_transfer

# Bake the checkpoints into the image (no runtime download, no volume).
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('ByteDance/LatentSync-1.5', local_dir='/app/LatentSync/checkpoints', \
allow_patterns=['latentsync_unet.pt','whisper/tiny.pt'])"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
