from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.pipeline import MangaTranslatorPipeline
from app.types import AppResult
from app.utils import ensure_dir, make_job_dir, resolve_ocr_lang, resolve_translation_lang, safe_text, sanitize_filename


def process_uploaded_image(
    filename,
    content,
    source_lang="en",
    translation_style="natural",
    performance_mode="balanced",
    debug_enabled=False,
    progress_callback=None,
) -> AppResult:
    translation_lang = resolve_translation_lang(source_lang)
    ocr_lang = resolve_ocr_lang(source_lang)
    style = safe_text(translation_style).lower()
    if style not in {"natural", "literal"}:
        style = "natural"
    mode = safe_text(performance_mode).lower()
    if mode not in {"quality", "balanced", "fast"}:
        mode = "balanced"
    config = _build_config(translation_lang, ocr_lang, style, mode, bool(debug_enabled))
    ensure_dir(config.output_dir)
    job_dir = make_job_dir(config.output_dir)

    safe_name = sanitize_filename(filename)
    input_path = Path(job_dir) / safe_name
    input_path.write_bytes(content)

    pipeline = MangaTranslatorPipeline(config=config)
    return pipeline.run(input_path, output_dir=job_dir, progress_callback=progress_callback)


def _build_config(source_lang: str, ocr_lang: str, translation_style: str, performance_mode: str, debug_enabled: bool) -> AppConfig:
    base = {
        "source_lang": source_lang,
        "target_lang": "pt",
        "ocr_lang": ocr_lang,
        "translation_style": translation_style,
        "performance_mode": performance_mode,
        "debug_enabled": debug_enabled,
    }
    if performance_mode == "fast":
        base.update(
            {
                "yolo_imgsz": 640,
                "yolo_confidence": 0.24,
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
                "yolo_imgsz": 1280,
                "yolo_confidence": 0.18,
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
                "max_render_font_attempts": 40,
            }
        )
    return AppConfig(**base)
