from __future__ import annotations

import gc
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from app.config import AppConfig
from app.pipeline import MangaTranslatorPipeline
from app.types import AppResult
from app.utils import (
    ensure_dir,
    get_translation_flow_config,
    make_job_dir,
    resolve_ocr_lang,
    resolve_translation_lang,
    resolve_translation_flow,
    safe_text,
    sanitize_filename,
)


_PIPELINE_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_active_job: dict = {"job_id": None, "started_at": None, "status": None}
_LOCK_TTL_SECONDS = int(os.getenv("PROCESSING_LOCK_TTL_SECONDS", "900") or "900")


def _force_release_if_expired() -> bool:
    with _STATE_LOCK:
        started = _active_job["started_at"]
        if started is None:
            return False
        age = time.time() - started
        if age < _LOCK_TTL_SECONDS:
            return False
        print(f"[LOCK] expired active_job={_active_job['job_id']} age_seconds={age:.1f}")
        _active_job["job_id"] = None
        _active_job["started_at"] = None
        _active_job["status"] = None
    if _PIPELINE_LOCK.locked():
        try:
            _PIPELINE_LOCK.release()
        except RuntimeError:
            pass
    return True


@contextmanager
def _single_page_processing(timeout_seconds: int = 30):
    job_id = uuid.uuid4().hex[:12]
    start = perf_counter()
    acquired = _PIPELINE_LOCK.acquire(timeout=max(1, int(timeout_seconds)))
    if not acquired and _force_release_if_expired():
        acquired = _PIPELINE_LOCK.acquire(blocking=False)
    wait_elapsed = perf_counter() - start
    if not acquired:
        with _STATE_LOCK:
            busy_id = _active_job["job_id"]
            busy_status = _active_job["status"]
            busy_started = _active_job["started_at"]
        busy_age = (time.time() - busy_started) if busy_started else 0.0
        print(f"[LOCK] busy active_job={busy_id} status={busy_status} age_seconds={busy_age:.1f}")
        raise RuntimeError("Outro processamento ainda esta em andamento. Tente novamente em instantes.")
    with _STATE_LOCK:
        _active_job["job_id"] = job_id
        _active_job["started_at"] = time.time()
        _active_job["status"] = "processing"
    print(f"[LOCK] acquire job_id={job_id}")
    print(f"[JOB] status job_id={job_id} status=processing")
    print(f"[FLOW] pipeline_lock_acquired wait={wait_elapsed:.2f}s")
    reason = "done"
    try:
        yield job_id
    except Exception as exc:
        reason = "failed"
        print(f"[JOB_ERROR] job_id={job_id} stage=pipeline error={exc}")
        raise
    finally:
        with _STATE_LOCK:
            _active_job["status"] = reason
            _active_job["job_id"] = None
            _active_job["started_at"] = None
        try:
            _PIPELINE_LOCK.release()
        except RuntimeError:
            pass
        print(f"[JOB] status job_id={job_id} status={reason}")
        print(f"[LOCK] release job_id={job_id} reason={reason}")
        print("[FLOW] pipeline_lock_released")


def reset_processing(reason: str = "cancelled") -> dict:
    with _STATE_LOCK:
        prev_id = _active_job["job_id"]
        prev_status = _active_job["status"]
        started = _active_job["started_at"]
        age = (time.time() - started) if started else 0.0
        _active_job["job_id"] = None
        _active_job["started_at"] = None
        _active_job["status"] = reason
    if _PIPELINE_LOCK.locked():
        try:
            _PIPELINE_LOCK.release()
        except RuntimeError:
            pass
    print(f"[LOCK] release job_id={prev_id} reason={reason}")
    print(f"[JOB] status job_id={prev_id} status={reason}")
    return {
        "released": True,
        "previous_job": prev_id,
        "previous_status": prev_status,
        "age_seconds": age,
        "reason": reason,
    }


def get_processing_status() -> dict:
    with _STATE_LOCK:
        started = _active_job["started_at"]
        return {
            "active_job": _active_job["job_id"],
            "status": _active_job["status"],
            "age_seconds": (time.time() - started) if started else 0.0,
            "ttl_seconds": _LOCK_TTL_SECONDS,
            "locked": _PIPELINE_LOCK.locked(),
        }


_DEFAULT_FLOW = os.getenv("DEFAULT_TRANSLATION_FLOW", "auto_to_en")
_DEFAULT_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "multilingual")
_DEFAULT_MODEL = os.getenv("TRANSLATION_MODEL") or os.getenv("HF_TRANSLATION_MODEL") or "facebook/m2m100_418M"


def process_uploaded_image(
    filename,
    content,
    translation_flow=None,
    source_lang=None,
    target_lang=None,
    ocr_lang=None,
    ocr_engine="auto",
    translation_style="natural",
    performance_mode="balanced",
    translation_provider=None,
    hf_translation_model=None,
    debug_enabled=False,
    progress_callback=None,
) -> AppResult:
    print(f"[FLOW] upload_received filename={safe_text(filename) or 'upload'} bytes={len(content or b'')}")
    # If the caller passed raw source/target codes but no flow, derive the
    # flow from the pair so OCR lang isn't silently inherited from the default
    # auto→en flow (which would pin OCR to Japanese).
    explicit_pair = bool(safe_text(source_lang)) and bool(safe_text(target_lang)) and not safe_text(translation_flow)
    if explicit_pair:
        candidate_flow = f"{resolve_translation_lang(source_lang)}_to_{resolve_translation_lang(target_lang)}"
        mode_key = resolve_translation_flow(candidate_flow)
    else:
        mode_key = resolve_translation_flow(translation_flow or _DEFAULT_FLOW)
    mode_config = get_translation_flow_config(mode_key)
    resolved_source_lang = resolve_translation_lang(source_lang or mode_config["source_lang"])
    resolved_target_lang = resolve_translation_lang(target_lang or mode_config["target_lang"])
    # When the caller forces a source different from the flow's, the flow's
    # OCR lang may not apply — fall back to source-derived OCR lang.
    if safe_text(source_lang) and resolved_source_lang != mode_config["source_lang"]:
        resolved_ocr_lang = safe_text(ocr_lang) or resolve_ocr_lang(resolved_source_lang)
    else:
        resolved_ocr_lang = safe_text(ocr_lang) or mode_config["ocr_lang"] or resolve_ocr_lang(resolved_source_lang)
    resolved_ocr_lang = resolve_ocr_lang(resolved_ocr_lang)
    print(
        f"[FLOW] lang_resolved source={resolved_source_lang} target={resolved_target_lang} "
        f"ocr={resolved_ocr_lang} flow={mode_key}"
    )
    resolved_ocr_engine = safe_text(ocr_engine).lower() or "auto"
    if resolved_ocr_engine not in {"auto", "paddle", "manga"}:
        resolved_ocr_engine = "auto"

    style = safe_text(translation_style).lower()
    if style not in {"natural", "literal"}:
        style = "natural"
    perf_mode = safe_text(performance_mode).lower()
    if perf_mode not in {"quality", "balanced", "fast"}:
        perf_mode = "balanced"
    provider = safe_text(translation_provider) or _DEFAULT_PROVIDER
    model = safe_text(hf_translation_model) or _DEFAULT_MODEL
    config = _build_config(
        translation_flow=mode_key,
        source_lang=resolved_source_lang,
        target_lang=resolved_target_lang,
        ocr_lang=resolved_ocr_lang,
        ocr_engine=resolved_ocr_engine,
        translation_style=style,
        performance_mode=perf_mode,
        translation_provider=provider,
        hf_translation_model=model,
        debug_enabled=bool(debug_enabled),
    )
    import tempfile
    import shutil
    
    # Use temporary directory for processing (auto-cleaned)
    with tempfile.TemporaryDirectory() as temp_dir:
        job_dir = Path(temp_dir)
        
        safe_name = sanitize_filename(filename)
        input_path = job_dir / safe_name
        try:
            input_path.write_bytes(content)

            print(f"[FLOW] processing_start job_dir={job_dir}")
            print(f"[FLOW] page_start file={safe_name}")
            with _single_page_processing(int(os.getenv("PIPELINE_LOCK_TIMEOUT_SECONDS", "30") or "30")):
                pipeline = MangaTranslatorPipeline(config=config)
                try:
                    result = pipeline.run(input_path, output_dir=None, progress_callback=progress_callback)
                finally:
                    # Drop the per-job pipeline (renderer/cleaner state) so its
                    # numpy buffers can be collected before the next page.
                    pipeline = None
                    _release_inference_memory()
            print(f"[FLOW] page_done file={safe_name}")
            print(f"[FLOW] processing_done job_dir={job_dir}")
            return result
        except Exception as exc:
            print(f"[FLOW_ERROR] processing {exc}")
            raise


def _release_inference_memory() -> None:
    gc.collect()
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_config(
    translation_flow: str,
    source_lang: str,
    target_lang: str,
    ocr_lang: str,
    ocr_engine: str,
    translation_style: str,
    performance_mode: str,
    translation_provider: str,
    hf_translation_model: str,
    debug_enabled: bool,
) -> AppConfig:
    base = {
        "translation_flow": resolve_translation_flow(translation_flow),
        "translation_provider": safe_text(translation_provider) or _DEFAULT_PROVIDER,
        "hf_translation_model": safe_text(hf_translation_model) or _DEFAULT_MODEL,
        "source_lang": safe_text(source_lang) or "auto",
        "target_lang": safe_text(target_lang) or "en",
        "ocr_lang": safe_text(ocr_lang) or "japan",
        "ocr_engine": ocr_engine,
        "translation_style": translation_style,
        "performance_mode": performance_mode,
        "debug_enabled": debug_enabled,
    }
    if performance_mode == "fast":
        base.update(
            {
                "yolo_imgsz": 512,
                "yolo_confidence": 0.26,
                "yolo_max_det": 25,
                "min_ocr_area": 1600,
                "min_ocr_width": 32,
                "min_ocr_height": 24,
                "min_ocr_dark_ratio": 0.002,
                "bubble_mask_close_px": 3,
                "bubble_mask_open_px": 1,
                "bubble_erode_px": 6,
                "text_mask_dilate_px": 7,
                "cleanup_morph_close_px": 3,
                "cleanup_extra_dilate_px": 1,
                "render_margin_px": 10,
                "max_font_size": 24,
                "font_shrink_step": 2,
                "max_render_font_attempts": 10,
            }
        )
    elif performance_mode == "quality":
        base.update(
            {
                "yolo_imgsz": 960,
                "yolo_confidence": 0.18,
                "yolo_max_det": 60,
                "min_ocr_area": 500,
                "min_ocr_width": 18,
                "min_ocr_height": 14,
                "min_ocr_dark_ratio": 0.0005,
                "bubble_mask_close_px": 6,
                "bubble_mask_open_px": 3,
                "text_mask_dilate_px": 12,
                "cleanup_morph_close_px": 5,
                "cleanup_extra_dilate_px": 3,
                "render_margin_px": 14,
                "max_font_size": 32,
                "max_render_font_attempts": 28,
            }
        )
    return AppConfig(**base)
