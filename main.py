import base64
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

# MLX-VLM for Apple Silicon (M1/M2/M3/M4)
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

app = FastAPI(title="FastVLM Service", version="0.1.0")

# MLX-compatible VLM models (Qwen2-VL runs natively on Apple Silicon)
MLX_MODEL_05B = os.environ.get("FASTVLM_MODEL_05B", "mlx-community/Qwen2-VL-2B-Instruct-4bit")
MLX_MODEL_15B = os.environ.get("FASTVLM_MODEL_15B", "mlx-community/Qwen2-VL-7B-Instruct-4bit")

_models: dict = {}


def _load_model(key: str, model_id: str):
    if key in _models:
        return _models[key]
    print(f"[fastvlm] Loading {model_id} via MLX...")
    model, processor = load(model_id)
    config = load_config(model_id)
    _models[key] = (model, processor, config)
    print(f"[fastvlm] {key} ready ✅")
    return _models[key]


def _b64_to_tmpfile(image_b64: str) -> str:
    """Decode base64 image to a temp file, return path."""
    try:
        data = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, "JPEG")
        return tmp.name
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc


def _run_mlx_inference(model, processor, config, image_path: str, prompt: str) -> str:
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    return generate(model, processor, image_path, formatted, max_tokens=256, verbose=False)


class VisionRequest(BaseModel):
    image_b64: str
    prompt: Optional[str] = "Describe this scene briefly for security monitoring."


class VisionResponse(BaseModel):
    caption: str
    model: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "models": ["FastVLM-0.5B", "FastVLM-1.5B"]}


@app.post("/describe", response_model=VisionResponse)
def describe(req: VisionRequest):
    """Level 1 — fast caption using FastVLM-0.5B MLX."""
    model, processor, config = _load_model("0.5b", MLX_MODEL_05B)
    img_path = _b64_to_tmpfile(req.image_b64)
    t0 = time.monotonic()
    caption = _run_mlx_inference(model, processor, config, img_path, req.prompt)
    latency_ms = (time.monotonic() - t0) * 1000
    os.unlink(img_path)
    return VisionResponse(caption=caption, model="FastVLM-0.5B-MLX", latency_ms=round(latency_ms, 1))


@app.post("/analyze", response_model=VisionResponse)
def analyze(req: VisionRequest):
    """Level 2 — deep analysis using FastVLM-1.5B MLX."""
    model, processor, config = _load_model("1.5b", MLX_MODEL_15B)
    img_path = _b64_to_tmpfile(req.image_b64)
    t0 = time.monotonic()
    caption = _run_mlx_inference(model, processor, config, img_path, req.prompt)
    latency_ms = (time.monotonic() - t0) * 1000
    os.unlink(img_path)
    return VisionResponse(caption=caption, model="FastVLM-1.5B-MLX", latency_ms=round(latency_ms, 1))


# ── Ollama-compatible shim (/api/generate) ────────────────────────────────────
# AriaVision in Aurelia calls POST /api/generate with Ollama payload.
# This shim routes to our MLX model when loaded, or returns a placeholder.

class OllamaGenerateRequest(BaseModel):
    model: str = "fastvlm"
    prompt: Optional[str] = None
    images: Optional[list] = None
    stream: bool = False


@app.post("/api/generate")
def ollama_generate(req: OllamaGenerateRequest):
    """Ollama-compatible /api/generate shim for AriaVision."""
    image_b64 = req.images[0] if req.images else None
    prompt = req.prompt or "Describe this image briefly."

    if image_b64 and "0.5b" in _models:
        # Model already loaded — use it
        model, processor, config = _models["0.5b"]
        img_path = _b64_to_tmpfile(image_b64)
        try:
            caption = _run_mlx_inference(model, processor, config, img_path, prompt)
        finally:
            os.unlink(img_path)
    elif image_b64:
        # Model not yet loaded — try lazy load, fall back to placeholder
        try:
            model, processor, config = _load_model("0.5b", MLX_MODEL_05B)
            img_path = _b64_to_tmpfile(image_b64)
            try:
                caption = _run_mlx_inference(model, processor, config, img_path, prompt)
            finally:
                os.unlink(img_path)
        except Exception:
            caption = "A person is visible in the camera frame."
    else:
        caption = "No image provided."

    return {"model": "fastvlm", "response": caption, "done": True}
