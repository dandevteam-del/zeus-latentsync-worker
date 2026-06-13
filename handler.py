"""RunPod serverless handler — LatentSync lip-sync on a rented CUDA GPU.

Deployed as a RunPod serverless endpoint (see README.md). Receives a base video
+ voice audio as base64, runs ByteDance LatentSync, returns the rendered mp4 as
base64. The Zeus Cut worker (modules/runpod_lipsync.py) is the client.

Input  (event["input"]):
    video_b64        base presenter clip (mp4), base64
    audio_b64        voiceover (wav), base64
    guidance_scale   default 1.5
    inference_steps  default 20
Output:
    video_b64        rendered lip-synced mp4, base64
"""
import base64
import os
import subprocess
import tempfile
import uuid

import runpod

import glob

LATENTSYNC_DIR = os.environ.get("LATENTSYNC_DIR", "/app/LatentSync")
UNET_CKPT = os.environ.get(
    "LATENTSYNC_UNET", f"{LATENTSYNC_DIR}/checkpoints/latentsync_unet.pt")


def _find_config() -> str:
    """LatentSync's unet config filename varies by version (stage2.yaml,
    stage2_512.yaml, ...). Honor an explicit override, else pick the best match."""
    override = os.environ.get("LATENTSYNC_CONFIG")
    if override and os.path.isfile(override):
        return override
    candidates = sorted(glob.glob(f"{LATENTSYNC_DIR}/configs/unet/*.yaml"))
    for pref in ("stage2", "unet"):
        for c in candidates:
            if pref in os.path.basename(c).lower():
                return c
    return candidates[0] if candidates else f"{LATENTSYNC_DIR}/configs/unet/stage2.yaml"


import time

CONFIG = _find_config()
HF_REPO = os.environ.get("LATENTSYNC_HF_REPO", "ByteDance/LatentSync-1.6")
# Checkpoints live on the network volume; LatentSync's code references them via
# the relative path "checkpoints/auxiliary" (from cwd /app/LatentSync), so we
# symlink /app/LatentSync/checkpoints -> the volume dir at startup.
VOL = "/runpod-volume"
CKPT_DIR = f"{VOL}/ls-ckpt"
REPO_CKPT_LINK = f"{LATENTSYNC_DIR}/checkpoints"

_weights_ready = False


def _dl(fn, what: str):
    last = None
    for attempt in range(3):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{what} download failed after retries: {last}")


def _ensure_weights() -> None:
    """Stage every model onto the network volume on first use (persistent, large
    — written once). Everything (unet, whisper, VAE via HF cache, insightface) is
    routed to /runpod-volume so the tiny container disk is never touched."""
    global _weights_ready
    if _weights_ready:
        return
    from huggingface_hub import snapshot_download

    # Make sure the volume dirs exist and the repo's relative checkpoints path
    # points at the volume (so the insightface root "checkpoints/auxiliary" and
    # the unet both resolve onto /runpod-volume).
    for d in (CKPT_DIR, f"{VOL}/hf/hub", f"{VOL}/torch", f"{VOL}/insightface", f"{VOL}/tmp"):
        os.makedirs(d, exist_ok=True)
    if not os.path.islink(REPO_CKPT_LINK):
        if os.path.isdir(REPO_CKPT_LINK):
            import shutil
            shutil.rmtree(REPO_CKPT_LINK, ignore_errors=True)
        try:
            os.symlink(CKPT_DIR, REPO_CKPT_LINK)
        except FileExistsError:
            pass

    # 1. unet + whisper → volume checkpoints
    if not os.path.isfile(UNET_CKPT):
        _dl(lambda: snapshot_download(
            HF_REPO, local_dir=CKPT_DIR,
            allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
            max_workers=4), "latentsync weights")

    # 2. VAE → HF cache on the volume (HF_HOME=/runpod-volume/hf)
    _dl(lambda: snapshot_download(
        "stabilityai/sd-vae-ft-mse",
        allow_patterns=["*.safetensors", "*.bin", "*.json"],
        max_workers=4), "sd-vae-ft-mse")

    # 3. insightface buffalo_l → volume (via the symlinked checkpoints/auxiliary)
    try:
        from insightface.app import FaceAnalysis
        FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"],
                     root=f"{CKPT_DIR}/auxiliary",
                     providers=["CPUExecutionProvider"]).prepare(ctx_id=-1, det_size=(640, 640))
    except Exception:
        pass  # LatentSync will fetch it itself onto the same volume path if needed

    _weights_ready = True


def _write_b64(b64: str, path: str) -> None:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def handler(event):
    inp = event.get("input") or {}
    if not inp.get("video_b64") or not inp.get("audio_b64"):
        return {"error": "video_b64 and audio_b64 are required"}

    try:
        _ensure_weights()
    except Exception as e:
        return {"error": f"weights unavailable: {e}"}

    work = tempfile.mkdtemp(prefix="latentsync-")
    tag = uuid.uuid4().hex[:8]
    video = os.path.join(work, f"in-{tag}.mp4")
    audio = os.path.join(work, f"in-{tag}.wav")
    out = os.path.join(work, f"out-{tag}.mp4")
    _write_b64(inp["video_b64"], video)
    _write_b64(inp["audio_b64"], audio)

    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", CONFIG,
        "--inference_ckpt_path", UNET_CKPT,
        "--video_path", video,
        "--audio_path", audio,
        "--video_out_path", out,
        "--guidance_scale", str(inp.get("guidance_scale", 1.5)),
        "--inference_steps", str(inp.get("inference_steps", 20)),
    ]
    proc = subprocess.run(cmd, cwd=LATENTSYNC_DIR, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(out):
        return {"error": "latentsync inference failed",
                "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-1000:]}

    return {"video_b64": _read_b64(out), "engine": "latentsync",
            "guidance_scale": inp.get("guidance_scale", 1.5),
            "inference_steps": inp.get("inference_steps", 20)}


runpod.serverless.start({"handler": handler})
