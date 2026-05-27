from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.backend import (
    get_processing_status,
    process_uploaded_image,
    reset_processing,
)
from app.utils import safe_text


app = FastAPI(title="Manga Translator Worker")
_worker_semaphore = asyncio.Semaphore(int(os.getenv("WORKER_CONCURRENCY", "1") or "1"))
_max_upload_mb = int(os.getenv("WORKER_MAX_UPLOAD_MB", "80") or "80")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "manga-translator-worker"}


@app.get("/api/processing/status")
def processing_status() -> dict:
    return get_processing_status()


@app.post("/api/processing/reset")
def processing_reset() -> dict:
    info = reset_processing(reason="cancelled")
    print(f"[LOCK] manual_reset previous_job={info['previous_job']} age_seconds={info['age_seconds']:.1f}")
    return info


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    translation_flow: str = Form(""),
    source_lang: str = Form(""),
    target_lang: str = Form(""),
    ocr_lang: str = Form(""),
    ocr_engine: str = Form("auto"),
    translation_style: str = Form("natural"),
    performance_mode: str = Form("balanced"),
    translation_provider: str = Form(""),
    translation_model: str = Form(""),
    debug_enabled: str = Form("false"),
) -> dict:
    os.environ["DISABLE_WORKER"] = "1"
    start = perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="arquivo vazio")
    if len(content) > _max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"arquivo maior que {_max_upload_mb}MB")

    print(f"[WORKER] request recebido arquivo={file.filename} bytes={len(content)}")
    if _worker_semaphore.locked():
        print("[WORKER] ocupado; rejeitando para evitar travar VPS pequena")
        raise HTTPException(status_code=503, detail="worker ocupado; tente novamente em instantes")

    async with _worker_semaphore:
        print("[WORKER] inicio processamento")
        try:
            result = await asyncio.to_thread(
                process_uploaded_image,
                filename=file.filename or "upload.png",
                content=content,
                translation_flow=safe_text(translation_flow) or None,
                source_lang=safe_text(source_lang) or None,
                target_lang=safe_text(target_lang) or None,
                ocr_lang=safe_text(ocr_lang) or None,
                ocr_engine=safe_text(ocr_engine) or "auto",
                translation_style=safe_text(translation_style) or "natural",
                performance_mode=safe_text(performance_mode) or "balanced",
                translation_provider=safe_text(translation_provider) or None,
                hf_translation_model=safe_text(translation_model) or None,
                debug_enabled=safe_text(debug_enabled).lower() in {"1", "true", "yes", "on"},
            )
        except Exception as exc:
            print(f"[WORKER] falha processamento: {exc}")
            raise HTTPException(status_code=500, detail=f"falha no processamento: {exc}") from exc

    translated_path = Path(result.translated_image_path)
    if not translated_path.exists():
        raise HTTPException(status_code=500, detail="imagem traduzida nao foi gerada")

    elapsed = perf_counter() - start
    print(f"[WORKER] fim processamento tempo={elapsed:.2f}s")
    return {
        "ok": True,
        "filename": translated_path.name,
        "image_base64": base64.b64encode(translated_path.read_bytes()).decode("ascii"),
        "metadata": result.metadata,
    }
