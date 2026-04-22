# fastvlm-service

FastAPI vision-language model service for the Nuclear ecosystem. Provides on-device scene captioning for nuclear-eye (alarm grading), nuclear-scout (mobile pedestrian detection), and the Sentinelle home security product.

Runs on Apple Silicon (M-series) via MLX, or as a CPU-mode fallback on Linux. Ships as a systemd service on b450.

---

## What it does

- Accepts base64-encoded images via HTTP POST
- Returns natural-language scene captions for security monitoring
- Two inference tiers:
  - `/describe` — fast caption (Qwen2-VL-2B, 4-bit MLX), Level 1
  - `/analyze` — deep analysis (Qwen2-VL-7B, 4-bit MLX), Level 2
- `/api/generate` — Ollama-compatible shim for AriaVision integration
- Lazy model loading — models load on first request

---

## Models

| Tier | Model | Default |
|------|-------|---------|
| 0.5B (fast) | `mlx-community/Qwen2-VL-2B-Instruct-4bit` | `FASTVLM_MODEL_05B` |
| 1.5B (deep) | `mlx-community/Qwen2-VL-7B-Instruct-4bit` | `FASTVLM_MODEL_15B` |

Models are downloaded from HuggingFace on first load and cached at `$HF_HOME` (default: `~/.cache/huggingface`).

---

## API

### `GET /health`

```json
{"status": "ok", "models": ["FastVLM-0.5B", "FastVLM-1.5B"]}
```

### `POST /describe`

Level 1 — fast caption via 0.5B model.

```json
{
  "image_b64": "<base64 JPEG or PNG>",
  "prompt": "Describe this scene briefly for security monitoring."
}
```

Response:
```json
{
  "caption": "A person is walking toward the front door.",
  "model": "FastVLM-0.5B-MLX",
  "latency_ms": 412.3
}
```

### `POST /analyze`

Level 2 — deep analysis via 1.5B model. Same request/response shape, higher latency.

### `POST /api/generate`

Ollama-compatible shim for AriaVision. Accepts standard Ollama payload with `images` list.

---

## Ports

| Service | Port |
|---------|------|
| fastvlm-service (standalone) | 8091 |
| fastvlm in Sentinelle stack | 8092 |

---

## Ecosystem integration

- **nuclear-eye / alarm-grader** — calls `/describe` for every camera frame graded as potential alarm
- **nuclear-scout** — iOS sensor app triggers `/describe` on pedestrian detection events
- **Sentinelle** — runs fastvlm as the `fastvlm` container in `docker-compose.yml` (port 8092)
- **Arianne / AriaVision** — uses `/api/generate` Ollama-compatible shim
- **vision-agent** — consumes `FASTVLM_URL` env var (default: `http://fastvlm:8092`)

---

## Security (bind + optional bearer)

- **`FASTVLM_BIND_HOST`** — default `127.0.0.1` when starting via `python main.py` (not `0.0.0.0`).
- **`FASTVLM_API_TOKEN`** — if set, `/describe`, `/analyze`, and `/api/generate` require `Authorization: Bearer <token>`; `/health` and OpenAPI paths stay open.

---

## Status

Runs as `fastvlm.service` systemd unit on b450. Port 8091. User: `crew`. Working directory: `/home/crew/git/fastvlm-service`.
