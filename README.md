# zeus-latentsync-worker

RunPod serverless worker for **ByteDance LatentSync** lip-sync — the GPU side of
the Zeus Cut HeyGen-parity avatar pipeline. RunPod builds this image from the
Dockerfile and runs `handler.py` as a serverless endpoint.

- **Input** (`event["input"]`): `video_b64`, `audio_b64`, `guidance_scale` (1.5), `inference_steps` (20)
- **Output**: `video_b64` — the rendered lip-synced mp4

LatentSync is OpenRAIL++ (commercial use OK with attribution). Weights are baked
at build time from `ByteDance/LatentSync-1.5`. Pay-per-second, scales to zero.

Client: `zeus/video-studio/worker/modules/runpod_lipsync.py`.
