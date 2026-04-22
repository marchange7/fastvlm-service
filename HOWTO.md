# HOWTO — fastvlm-service

Setup, model download, running standalone, systemd deployment on b450, and integration with Sentinelle and nuclear-eye.

---

## Table of contents

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Download models](#3-download-models)
4. [Run locally (development)](#4-run-locally-development)
5. [Test the API](#5-test-the-api)
6. [Deploy as systemd service on b450](#6-deploy-as-systemd-service-on-b450)
7. [Run inside Sentinelle (Docker)](#7-run-inside-sentinelle-docker)
8. [Configure models via env vars](#8-configure-models-via-env-vars)
9. [Integrate with nuclear-eye](#9-integrate-with-nuclear-eye)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Requirements

| Platform | Runtime | Notes |
|----------|---------|-------|
| Apple Silicon (M1–M4) | MLX via mlx-vlm | Full GPU acceleration |
| Linux x86 (b450, Thunder) | CPU fallback | Slow but functional |
| Linux + NVIDIA GPU | CUDA via PyTorch | Set `torch` with CUDA wheels |
| Jetson Orin | ARM CPU fallback | Functional at ~2 fps |

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for 1.5B model)
- ~8 GB disk for both models

---

## 2. Install

### Create a virtual environment

```bash
cd ~/git/fastvlm-service   # or /data/git/fastvlm-service on b450
python3 -m venv ~/.venv/nuclear
source ~/.venv/nuclear/bin/activate
pip install -e .
```

### On Apple Silicon — install MLX

```bash
pip install mlx-vlm
```

The base `pyproject.toml` installs `torch` + `transformers`. MLX replaces those for inference on Apple Silicon.

---

## 3. Download models

```bash
bash download_models.sh
```

This downloads:
- `mlx-community/Qwen2-VL-2B-Instruct-4bit` → `~/models/fastvlm-0.5b`
- `mlx-community/Qwen2-VL-7B-Instruct-4bit` → `~/models/fastvlm-1.5b`

Models are cached at `$HF_HOME` (default: `~/.cache/huggingface`). On b450, point this to `/data`:

```bash
export HF_HOME=/data/.cache/huggingface
bash download_models.sh
```

---

## 4. Run locally (development)

```bash
source ~/.venv/nuclear/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8091 --reload
```

Service starts at `http://localhost:8091`.

Models load lazily on first request — expect 15–60s on first call while model loads.

---

## 5. Test the API

### Health check

```bash
curl http://localhost:8091/health
```

Expected:
```json
{"status": "ok", "models": ["FastVLM-0.5B", "FastVLM-1.5B"]}
```

### Send an image for Level 1 caption

```bash
# Encode a test image
B64=$(base64 -i ~/Pictures/test.jpg)

curl -s -X POST http://localhost:8091/describe \
    -H 'Content-Type: application/json' \
    -d "{\"image_b64\": \"$B64\"}" | jq
```

Expected:
```json
{
  "caption": "A person is standing near the front door.",
  "model": "FastVLM-0.5B-MLX",
  "latency_ms": 412.3
}
```

### Send for Level 2 deep analysis

```bash
curl -s -X POST http://localhost:8091/analyze \
    -H 'Content-Type: application/json' \
    -d "{\"image_b64\": \"$B64\", \"prompt\": \"Is there any suspicious activity?\"}" | jq
```

### Ollama-compatible shim (AriaVision)

```bash
curl -s -X POST http://localhost:8091/api/generate \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"fastvlm\", \"prompt\": \"What is in this image?\", \"images\": [\"$B64\"]}" | jq
```

---

## 6. Deploy as systemd service on b450

### Copy the service file

```bash
ssh b450
sudo cp /data/git/fastvlm-service/fastvlm.service /etc/systemd/system/fastvlm.service
```

The service file configures:
- User: `crew`
- WorkingDirectory: `/home/crew/git/fastvlm-service`
- ExecStart: `uvicorn main:app --host 0.0.0.0 --port 8091`
- HF_HOME: `/home/crew/.cache/huggingface`

### Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable fastvlm.service
sudo systemctl start fastvlm.service
```

### Check status

```bash
sudo systemctl status fastvlm.service
journalctl -u fastvlm.service -f
```

### Upgrade binary (Nuclear deploy pattern)

```bash
cargo build --release   # not applicable — Python service; pull new code instead
sudo systemctl stop fastvlm.service
cd /data/git/fastvlm-service && git pull
sudo systemctl start fastvlm.service
```

---

## 7. Run inside Sentinelle (Docker)

fastvlm-service is built and managed by Sentinelle's `docker-compose.yml`. You do not need to run it manually when using Sentinelle.

```bash
# From the sentinelle repo:
docker compose up -d fastvlm
docker compose logs -f fastvlm
```

The Sentinelle compose file builds from `../fastvlm-service` and exposes port 8092 on `127.0.0.1`.

vision-agent reads `FASTVLM_URL` (default: `http://fastvlm:8092`).

To disable fastvlm in cloud/resource-constrained deployments:
```bash
FASTVLM_DISABLED=true VISION_CLOUD_FALLBACK=true docker compose up -d
```

---

## 8. Configure models via env vars

Override which model each tier loads:

```bash
export FASTVLM_MODEL_05B="mlx-community/Qwen2-VL-2B-Instruct-4bit"   # Level 1 (default)
export FASTVLM_MODEL_15B="mlx-community/Qwen2-VL-7B-Instruct-4bit"   # Level 2 (default)
```

Set in `.env` for Sentinelle, or in the systemd service `Environment=` lines for standalone.

---

## 9. Integrate with nuclear-eye

nuclear-eye's vision-agent calls fastvlm via `FASTVLM_URL`. Standalone:

```bash
# In nuclear-eye .env or systemd environment:
FASTVLM_URL=http://127.0.0.1:8091
```

In Sentinelle docker-compose the service name resolves automatically:
```
FASTVLM_URL=http://fastvlm:8092
```

AriaVision (Arianne) uses the Ollama-compatible shim at `/api/generate`. Point it to the same host:
```bash
OLLAMA_BASE_URL=http://127.0.0.1:8091
```

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| First request hangs | Normal — model is loading. Wait 15–60s. Check logs for `[fastvlm] Loading...` |
| `ModuleNotFoundError: mlx_vlm` | `pip install mlx-vlm` (Apple Silicon only) |
| `torch` import error on Mac | Use MLX path; torch is for Linux. Install `mlx-vlm` instead |
| OOM / killed on b450 | 1.5B model needs ~6 GB RAM. Use only `/describe` (0.5B) on memory-limited nodes |
| `HF_HOME` filling `/` on b450 | `export HF_HOME=/data/.cache/huggingface` and restart service |
| Port 8091 already in use | Check `sudo ss -tlnp | grep 8091`; another service may have the port |
| `FASTVLM_DISABLED=true` set in Sentinelle | Expected on cloud profile; `vision-agent` falls back to `VISION_CLOUD_FALLBACK` path |
| systemd service fails to start | `journalctl -u fastvlm.service -n 50`; check venv path and HF_HOME |
