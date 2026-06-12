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

LATENTSYNC_DIR = os.environ.get("LATENTSYNC_DIR", "/app/LatentSync")
UNET_CKPT = os.environ.get(
    "LATENTSYNC_UNET", f"{LATENTSYNC_DIR}/checkpoints/latentsync_unet.pt")
CONFIG = os.environ.get(
    "LATENTSYNC_CONFIG", f"{LATENTSYNC_DIR}/configs/unet/stage2.yaml")


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
