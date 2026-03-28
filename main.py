import base64
import io
import os
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

app = FastAPI(title="FastVLM Service", version="0.1.0")

MODEL_DIR_05B = os.path.expanduser("~/models/fastvlm-0.5b")
MODEL_DIR_15B = os.path.expanduser("~/models/fastvlm-1.5b")
HF_MODEL_05B = "apple/FastVLM-0.5B-fp16"
HF_MODEL_15B = "apple/FastVLM-1.5B-int8"

_models: dict = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def _load_model(key: str, model_dir: str, hf_repo: str):
    if key in _models:
        return _models[key]

    local_path = model_dir if os.path.isdir(model_dir) else hf_repo
    processor = AutoProcessor.from_pretrained(local_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        local_path,
        torch_dtype=TORCH_DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        model = model.to(DEVICE)
    model.eval()
    _models[key] = (processor, model)
    return _models[key]


def _b64_to_image(image_b64: str) -> Image.Image:
    try:
        data = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64: {exc}") from exc


def _run_inference(processor, model, image: Image.Image, prompt: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


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
    """Level 1 — fast caption using FastVLM-0.5B-fp16."""
    processor, model = _load_model("0.5b", MODEL_DIR_05B, HF_MODEL_05B)
    image = _b64_to_image(req.image_b64)
    t0 = time.monotonic()
    caption = _run_inference(processor, model, image, req.prompt)
    latency_ms = (time.monotonic() - t0) * 1000
    return VisionResponse(caption=caption, model="FastVLM-0.5B", latency_ms=round(latency_ms, 1))


@app.post("/analyze", response_model=VisionResponse)
def analyze(req: VisionRequest):
    """Level 2 — deep analysis using FastVLM-1.5B-int8."""
    processor, model = _load_model("1.5b", MODEL_DIR_15B, HF_MODEL_15B)
    image = _b64_to_image(req.image_b64)
    t0 = time.monotonic()
    caption = _run_inference(processor, model, image, req.prompt)
    latency_ms = (time.monotonic() - t0) * 1000
    return VisionResponse(caption=caption, model="FastVLM-1.5B", latency_ms=round(latency_ms, 1))
