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
    progress_callback=None,
) -> AppResult:
    translation_lang = resolve_translation_lang(source_lang)
    ocr_lang = resolve_ocr_lang(source_lang)
    style = safe_text(translation_style).lower()
    if style not in {"natural", "literal"}:
        style = "natural"
    config = AppConfig(source_lang=translation_lang, target_lang="pt", ocr_lang=ocr_lang, translation_style=style)
    ensure_dir(config.output_dir)
    job_dir = make_job_dir(config.output_dir)

    safe_name = sanitize_filename(filename)
    input_path = Path(job_dir) / safe_name
    input_path.write_bytes(content)

    pipeline = MangaTranslatorPipeline(config=config)
    return pipeline.run(input_path, output_dir=job_dir, progress_callback=progress_callback)
