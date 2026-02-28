#!/usr/bin/env python3
import base64
import io
import os
import threading
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor


def _dtype_from_env() -> torch.dtype:
    value = os.getenv("COLQWEN_DTYPE", "bfloat16").lower()
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    return torch.bfloat16


MODEL_ID = os.getenv("MODEL_ID", "TomoroAI/tomoro-ai-colqwen3-embed-4b-awq")
MAX_NUM_VISUAL_TOKENS = int(os.getenv("COLQWEN_MAX_NUM_VISUAL_TOKENS", "1280"))
ATTN_IMPL = os.getenv("COLQWEN_ATTN_IMPLEMENTATION", "sdpa")
DTYPE = _dtype_from_env()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRELOAD_ON_START = os.getenv("COLQWEN_PRELOAD_ON_START", "1") == "1"

app = FastAPI(title="ColQwen3 Embed Server", version="0.1.0")

_init_lock = threading.Lock()
_processor = None
_model = None


class PoolingRequest(BaseModel):
    model: str | None = None
    task: str | None = None
    input: Any | None = None
    messages: list[dict[str, Any]] | None = None


def _image_from_data_url(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("Only data URL images are supported in this server")
    try:
        b64 = url.split(",", 1)[1]
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image data URL: {exc}") from exc


def _extract_texts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            text = " ".join([p for p in parts if p.strip()]).strip()
            if text:
                texts.append(text)
    return texts


def _extract_images_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url", {})
            if not isinstance(image_url, dict) or "url" not in image_url:
                continue
            images.append(_image_from_data_url(image_url["url"]))
    return images


def _ensure_loaded() -> tuple[Any, Any]:
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model
    with _init_lock:
        if _processor is not None and _model is not None:
            return _processor, _model
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            max_num_visual_tokens=MAX_NUM_VISUAL_TOKENS,
        )
        _model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=DTYPE,
            attn_implementation=ATTN_IMPL,
            trust_remote_code=True,
            device_map=DEVICE,
        ).eval()
        return _processor, _model


def _to_vectors(tensor: torch.Tensor) -> list[list[float]]:
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Unexpected embeddings ndim={tensor.ndim}")
    return tensor.to(torch.float32).cpu().tolist()


@app.on_event("startup")
def startup_preload() -> None:
    if not PRELOAD_ON_START:
        print("[startup] colqwen preload disabled", flush=True)
        return
    print("[startup] colqwen preload start", flush=True)
    _ensure_loaded()
    print("[startup] colqwen preload done", flush=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "colqwen3-embed-server",
            }
        ],
    }


@app.post("/pooling")
def pooling(req: PoolingRequest) -> dict[str, Any]:
    if req.task and req.task != "token_embed":
        raise HTTPException(status_code=400, detail="Only task=token_embed is supported")

    processor, model = _ensure_loaded()

    texts: list[str] = []
    images: list[Image.Image] = []

    if req.messages:
        texts = _extract_texts_from_messages(req.messages)
        images = _extract_images_from_messages(req.messages)
    elif req.input is not None:
        if isinstance(req.input, str):
            texts = [req.input]
        elif isinstance(req.input, list):
            texts = [x for x in req.input if isinstance(x, str)]

    if images:
        features = processor.process_images(images=images)
        features = {
            k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        with torch.inference_mode():
            out = model(**features)
        vectors = _to_vectors(out.embeddings)
    elif texts:
        batch = processor.process_texts(texts=texts)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**batch)
        vectors = _to_vectors(out.embeddings)
    else:
        raise HTTPException(status_code=400, detail="No text or image input found")

    return {
        "object": "list",
        "data": [
            {
                "object": "pooling",
                "index": 0,
                "data": vectors,
            }
        ],
    }
