import base64
import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from adapters import EditGuardAdapter, StegaStampAdapter
from services.pipeline import random_bits_64, run_embed_pipeline
from services.verify import run_verify_pipeline


DEFAULT_EDITGUARD_ROOT = Path("/home/project/Documents/EditGuard")
DEFAULT_STEGASTAMP_ROOT = Path("/home/project/Documents/StegaStamp-pytorch")
DEFAULT_STEGASTAMP_MODEL_PATH = DEFAULT_STEGASTAMP_ROOT / "asset" / "best.pth"
DEFAULT_STEGASTAMP_ENV = "unused"

_ADAPTER_CACHE: Dict[Tuple[str, str, str], Tuple[EditGuardAdapter, StegaStampAdapter]] = {}


class EmbedRequest(BaseModel):
    image_base64: str
    editguard_bits: str
    stegastamp_secret: str
    editguard_root: str = str(DEFAULT_EDITGUARD_ROOT)
    stegastamp_root: str = str(DEFAULT_STEGASTAMP_ROOT)
    stegastamp_env: str = DEFAULT_STEGASTAMP_ENV
    stegastamp_model_path: str = str(DEFAULT_STEGASTAMP_MODEL_PATH)


class InpaintRequest(BaseModel):
    image_base64: str
    mask_base64: str
    prompt: str = "repair tampered region"
    editguard_root: str = str(DEFAULT_EDITGUARD_ROOT)
    stegastamp_root: str = str(DEFAULT_STEGASTAMP_ROOT)
    stegastamp_env: str = DEFAULT_STEGASTAMP_ENV


class VerifyRequest(BaseModel):
    image_base64: str
    metadata_json: str
    editguard_root: str = str(DEFAULT_EDITGUARD_ROOT)
    stegastamp_root: str = str(DEFAULT_STEGASTAMP_ROOT)
    stegastamp_env: str = DEFAULT_STEGASTAMP_ENV
    stegastamp_model_path: str = str(DEFAULT_STEGASTAMP_MODEL_PATH)


def _strip_data_url_prefix(raw: str) -> str:
    if "," in raw and raw.split(",", 1)[0].startswith("data:image"):
        return raw.split(",", 1)[1]
    return raw


def _decode_base64_image(image_base64: str) -> np.ndarray:
    try:
        content = base64.b64decode(_strip_data_url_prefix(image_base64))
    except Exception as exc:
        raise ValueError("Invalid base64 image input.") from exc

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise ValueError("Failed to decode image content.") from exc

    return np.array(image, dtype=np.uint8)


def _decode_base64_mask(mask_base64: str) -> np.ndarray:
    try:
        content = base64.b64decode(_strip_data_url_prefix(mask_base64))
    except Exception as exc:
        raise ValueError("Invalid base64 mask input.") from exc

    try:
        mask = Image.open(io.BytesIO(content)).convert("L")
    except Exception as exc:
        raise ValueError("Failed to decode mask content.") from exc

    return np.array(mask, dtype=np.uint8)


def _encode_png_base64(image_np: np.ndarray) -> str:
    image_np = np.array(image_np)
    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating) and np.max(image_np) <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    if image_np.ndim == 2:
        image = Image.fromarray(image_np, mode="L")
    else:
        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]
        image = Image.fromarray(image_np, mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _get_adapters(editguard_root: str, stegastamp_root: str, stegastamp_env: str):
    key = (str(Path(editguard_root).resolve()), str(Path(stegastamp_root).resolve()), stegastamp_env)
    cached = _ADAPTER_CACHE.get(key)
    if cached is not None:
        return cached

    editguard_adapter = EditGuardAdapter(editguard_root=editguard_root)
    stegastamp_adapter = StegaStampAdapter(stegastamp_root=stegastamp_root, stegastamp_env=stegastamp_env)
    _ADAPTER_CACHE[key] = (editguard_adapter, stegastamp_adapter)
    return editguard_adapter, stegastamp_adapter


app = FastAPI(title="Dual-WaterMark API", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/random-bits")
def random_bits():
    return {"bits": random_bits_64()}


@app.post("/embed")
def embed(request: EmbedRequest):
    try:
        image = _decode_base64_image(request.image_base64)
        editguard_adapter, stegastamp_adapter = _get_adapters(
            request.editguard_root,
            request.stegastamp_root,
            request.stegastamp_env,
        )

        result = run_embed_pipeline(
            image_rgb_uint8=image,
            editguard_bits=request.editguard_bits,
            stegastamp_secret=request.stegastamp_secret,
            editguard_adapter=editguard_adapter,
            stegastamp_adapter=stegastamp_adapter,
            stegastamp_model_dir=request.stegastamp_model_path,
        )

        return {
            "metadata_json": result["metadata_json"],
            "stegastamp_image_base64": _encode_png_base64(result.get("stegastamp_image", result["final_image"])),
            "final_image_base64": _encode_png_base64(result["final_image"]),
            "stegastamp_residual_base64": _encode_png_base64(result["stegastamp_residual"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/inpaint")
def inpaint(request: InpaintRequest):
    try:
        image = _decode_base64_image(request.image_base64)
        mask = _decode_base64_mask(request.mask_base64)

        editguard_adapter, _ = _get_adapters(
            request.editguard_root,
            request.stegastamp_root,
            request.stegastamp_env,
        )

        output = editguard_adapter.inpaint(
            image=image,
            prompt=request.prompt,
            mask=mask,
        )

        return {
            "inpainted_image_base64": _encode_png_base64(output),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/verify")
def verify(request: VerifyRequest):
    try:
        image = _decode_base64_image(request.image_base64)
        editguard_adapter, stegastamp_adapter = _get_adapters(
            request.editguard_root,
            request.stegastamp_root,
            request.stegastamp_env,
        )

        result = run_verify_pipeline(
            image_rgb_uint8=image,
            metadata_json=request.metadata_json,
            editguard_adapter=editguard_adapter,
            stegastamp_adapter=stegastamp_adapter,
            detector_model_dir="",
            stegastamp_model_dir=request.stegastamp_model_path,
        )

        return {
            "stegastamp_found_codes": result["stegastamp_found_codes"],
            "editguard_recovered_bits": result["editguard_recovered_bits"],
            "editguard_accuracy": result["editguard_accuracy"],
            "editguard_mask_base64": _encode_png_base64(result["editguard_mask"]),
            "summary": result["summary"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
