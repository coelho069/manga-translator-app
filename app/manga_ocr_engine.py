from __future__ import annotations

import os
import sys

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

_MODEL_NAME = "kha-white/manga-ocr-base"
_CACHE_DIR = "models/huggingface"
_DEVICE = "cpu"
_TOKENIZER: Any | None = None
_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_MODEL_DTYPE: Any | None = None
_LOCK = Lock()


def configure_manga_ocr(model_name: str | None = None, cache_dir: str | None = None, device: str | None = None) -> None:
    global _MODEL_NAME, _CACHE_DIR, _DEVICE
    if model_name:
        _MODEL_NAME = model_name
    if cache_dir:
        _CACHE_DIR = cache_dir
    if device:
        _DEVICE = device


def run_manga_ocr(image_or_crop) -> str | None:
    try:
        image = _to_pil_image(image_or_crop)
        if image is None:
            return None
        tokenizer, model, processor = _load_model()
        _log("[MANGA-OCR] reconhecendo texto")
        text = _generate_text(image, tokenizer, model, processor)
        text = " ".join(str(text or "").split())
        if text:
            _log(f"[MANGA-OCR] texto reconhecido: {text}")
            return text
    except Exception as exc:
        _log(f"[MANGA-OCR] falha, usando fallback: {exc}")
    return None


def _load_model():
    global _TOKENIZER, _MODEL, _PROCESSOR, _MODEL_DTYPE
    with _LOCK:
        if _TOKENIZER is not None and _MODEL is not None:
            return _TOKENIZER, _MODEL, _PROCESSOR

        _log("[MODEL] Manga OCR carregando uma vez")
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer

        try:
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        except ImportError:
            from transformers import VisionEncoderDecoderModel
            model_class = VisionEncoderDecoderModel

        try:
            _PROCESSOR = AutoImageProcessor.from_pretrained(_MODEL_NAME, cache_dir=_CACHE_DIR)
        except Exception:
            _PROCESSOR = None

        dtype_env = (os.getenv("MANGA_OCR_DTYPE", "float32") or "float32").lower()
        load_kwargs = {"low_cpu_mem_usage": True}
        if dtype_env in {"float16", "fp16", "half"}:
            load_kwargs["torch_dtype"] = torch.float16
            _MODEL_DTYPE = torch.float16
        elif dtype_env in {"bfloat16", "bf16"}:
            load_kwargs["torch_dtype"] = torch.bfloat16
            _MODEL_DTYPE = torch.bfloat16
        else:
            _MODEL_DTYPE = torch.float32

        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir=_CACHE_DIR)
        _MODEL = model_class.from_pretrained(_MODEL_NAME, cache_dir=_CACHE_DIR, **load_kwargs)
        if hasattr(_MODEL, "to"):
            _MODEL = _MODEL.to(_DEVICE)
        if hasattr(_MODEL, "eval"):
            _MODEL.eval()
        _log(f"[MODEL] Manga OCR carregado uma vez dtype={dtype_env}")
        return _TOKENIZER, _MODEL, _PROCESSOR


def _generate_text(image: Image.Image, tokenizer, model, processor) -> str | None:
    import torch

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    image = image.convert("RGB")
    _log(f"[MANGA-OCR] tamanho da imagem para modelo: {image.size[0]}x{image.size[1]}")
    if processor is not None:
        try:
            start = perf_counter()
            inputs = processor(images=image, return_tensors="pt")
            inputs = _move_inputs(inputs, _DEVICE)
            if _MODEL_DTYPE is not None and _MODEL_DTYPE != torch.float32:
                inputs = _cast_float_inputs(inputs, _MODEL_DTYPE)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=180,
                    num_beams=1,
                    early_stopping=False,
                    max_time=45,
                )
            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            _log(f"[MANGA-OCR] tokens gerados: {output_ids.shape[-1]}")
            _log(f"[PERF] manga_ocr_generate={perf_counter() - start:.4f}s")
            return text
        except Exception as exc:
            _log(f"[MANGA-OCR] falha na geracao: {exc}")

    raise RuntimeError("classe do Transformers nao conseguiu processar imagem")


def _move_inputs(inputs, device: str):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _cast_float_inputs(inputs, dtype):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(dtype=dtype)
        except Exception:
            return inputs
    cast: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "dtype") and getattr(value, "is_floating_point", lambda: False)():
            try:
                cast[key] = value.to(dtype=dtype)
                continue
            except Exception:
                pass
        cast[key] = value
    return cast


def _to_pil_image(image_or_crop) -> Image.Image | None:
    if image_or_crop is None:
        return None
    if isinstance(image_or_crop, Image.Image):
        return image_or_crop.convert("RGB")
    if isinstance(image_or_crop, (str, Path)):
        return Image.open(image_or_crop).convert("RGB")
    if isinstance(image_or_crop, np.ndarray):
        if image_or_crop.size == 0:
            return None
        if image_or_crop.ndim == 2:
            return Image.fromarray(image_or_crop).convert("RGB")
        if image_or_crop.shape[2] == 4:
            # BGRA do OpenCV → RGB (descarta canal alpha, inverte BGR→RGB)
            return Image.fromarray(image_or_crop[:, :, 2::-1]).convert("RGB")
        if image_or_crop.shape[2] == 3:
            return Image.fromarray(image_or_crop[:, :, ::-1]).convert("RGB")
        return Image.fromarray(image_or_crop).convert("RGB")
    return None


def _log(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_message = str(message).encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
    print(safe_message)
