from __future__ import annotations

import zipfile
from pathlib import Path
from time import perf_counter

from app.config import AppConfig
from app.pipeline import MangaTranslatorPipeline
from app.types import AppResult
from app.utils import (
    ensure_dir,
    get_translation_mode_config,
    make_job_dir,
    resolve_ocr_lang,
    resolve_translation_lang,
    resolve_translation_mode,
    safe_text,
    sanitize_filename,
)


def process_uploaded_image(
    filename,
    content,
    translation_mode="en_to_pt",
    source_lang=None,
    target_lang=None,
    ocr_lang=None,
    translation_style="natural",
    performance_mode="balanced",
    debug_enabled=False,
    progress_callback=None,
    min_font_size=None,
    max_font_size=None,
    line_spacing_ratio=None,
    auto_font_resize=None,
    center_text=None,
    bold_text=None,
    text_color=None,
) -> AppResult:
    mode_key = resolve_translation_mode(translation_mode)
    mode_config = get_translation_mode_config(mode_key)
    resolved_source_lang = resolve_translation_lang(source_lang or mode_config["source_lang"])
    resolved_target_lang = resolve_translation_lang(target_lang or mode_config["target_lang"])
    resolved_ocr_lang = safe_text(ocr_lang) or mode_config["ocr_lang"] or resolve_ocr_lang(resolved_source_lang)
    resolved_ocr_lang = resolve_ocr_lang(resolved_ocr_lang)

    style = safe_text(translation_style).lower()
    if style not in {"natural", "literal"}:
        style = "natural"
    perf_mode = safe_text(performance_mode).lower()
    if perf_mode not in {"quality", "balanced", "fast"}:
        perf_mode = "balanced"
    config = _build_config(
        translation_mode=resolve_translation_mode(translation_mode),
        source_lang=resolved_source_lang,
        target_lang=resolved_target_lang,
        ocr_lang=resolved_ocr_lang,
        translation_style=style,
        performance_mode=perf_mode,
        debug_enabled=bool(debug_enabled),
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        line_spacing_ratio=line_spacing_ratio,
        auto_font_resize=auto_font_resize,
        center_text=center_text,
        bold_text=bold_text,
        text_color=text_color,
    )
    ensure_dir(config.output_dir)
    job_dir = make_job_dir(config.output_dir)

    safe_name = sanitize_filename(filename)
    input_path = Path(job_dir) / safe_name
    input_path.write_bytes(content)

    pipeline = MangaTranslatorPipeline(config=config)
    return pipeline.run(input_path, output_dir=job_dir, progress_callback=progress_callback)


def _build_config(
    translation_mode: str,
    source_lang: str,
    target_lang: str,
    ocr_lang: str,
    translation_style: str,
    performance_mode: str,
    debug_enabled: bool,
    min_font_size: int | None = None,
    max_font_size: int | None = None,
    line_spacing_ratio: float | None = None,
    auto_font_resize: bool | None = None,
    center_text: bool | None = None,
    bold_text: bool | None = None,
    text_color: tuple[int, int, int] | None = None,
) -> AppConfig:
    base = {
        "translation_mode": resolve_translation_mode(translation_mode),
        "source_lang": source_lang,
        "target_lang": target_lang,
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
    if min_font_size is not None:
        base["min_font_size"] = int(min_font_size)
    if max_font_size is not None:
        base["max_font_size"] = int(max_font_size)
    if line_spacing_ratio is not None:
        base["line_spacing_ratio"] = float(line_spacing_ratio)
    if auto_font_resize is not None:
        base["auto_font_resize"] = bool(auto_font_resize)
    if center_text is not None:
        base["center_text"] = bool(center_text)
    if bold_text is not None:
        base["bold_text"] = bool(bold_text)
    if text_color is not None:
        base["text_color"] = tuple(text_color)

    return AppConfig(**base)


def process_batch(
    files: list[tuple[str, bytes]],
    translation_mode="en_to_pt",
    source_lang=None,
    target_lang=None,
    ocr_lang=None,
    translation_style="natural",
    performance_mode="balanced",
    debug_enabled=False,
    progress_callback=None,
    min_font_size=None,
    max_font_size=None,
    line_spacing_ratio=None,
    auto_font_resize=None,
    center_text=None,
    bold_text=None,
    text_color=None,
) -> list[AppResult]:
    mode_key = resolve_translation_mode(translation_mode)
    mode_config = get_translation_mode_config(mode_key)
    resolved_source_lang = resolve_translation_lang(source_lang or mode_config["source_lang"])
    resolved_target_lang = resolve_translation_lang(target_lang or mode_config["target_lang"])
    resolved_ocr_lang = safe_text(ocr_lang) or mode_config["ocr_lang"] or resolve_ocr_lang(resolved_source_lang)
    resolved_ocr_lang = resolve_ocr_lang(resolved_ocr_lang)

    style = safe_text(translation_style).lower()
    if style not in {"natural", "literal"}:
        style = "natural"
    perf_mode = safe_text(performance_mode).lower()
    if perf_mode not in {"quality", "balanced", "fast"}:
        perf_mode = "balanced"

    config = _build_config(
        translation_mode=resolve_translation_mode(translation_mode),
        source_lang=resolved_source_lang,
        target_lang=resolved_target_lang,
        ocr_lang=resolved_ocr_lang,
        translation_style=style,
        performance_mode=perf_mode,
        debug_enabled=bool(debug_enabled),
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        line_spacing_ratio=line_spacing_ratio,
        auto_font_resize=auto_font_resize,
        center_text=center_text,
        bold_text=bold_text,
        text_color=text_color,
    )
    ensure_dir(config.output_dir)

    pipeline = MangaTranslatorPipeline(config=config)

    results: list[AppResult] = []
    total_files = len(files)
    batch_start = perf_counter()
    page_times: list[float] = []

    print(f"\n{'='*60}")
    print(f"[BATCH] Iniciando lote de {total_files} paginas")
    print(f"[BATCH] Modo: {perf_mode} | Traducao: {style}")
    print(f"{'='*60}\n")

    for idx, (filename, content) in enumerate(files):
        page_start = perf_counter()

        def page_progress(value: float, message: str, meta=None) -> None:
            if progress_callback is not None:
                overall = (idx + value) / total_files
                try:
                    progress_callback(overall, f"Pagina {idx + 1}/{total_files}: {message}", meta)
                except TypeError:
                    progress_callback(overall, f"Pagina {idx + 1}/{total_files}: {message}")

        job_dir = make_job_dir(config.output_dir)
        safe_name = sanitize_filename(filename)
        input_path = Path(job_dir) / safe_name
        input_path.write_bytes(content)

        try:
            result = pipeline.run(input_path, output_dir=job_dir, progress_callback=page_progress)
            results.append(result)
        except Exception as exc:
            print(f"[BATCH] ERRO na pagina {idx + 1} ({filename}): {exc}")
            continue

        page_elapsed = round(perf_counter() - page_start, 2)
        page_times.append(page_elapsed)

        bubbles = result.bubbles
        timings = result.metadata.get("timings", {})
        detected = result.metadata.get("detected_count", 0)
        rendered = result.metadata.get("rendered_count", 0)

        print(f"[BATCH] Pagina {idx + 1}/{total_files} concluida em {page_elapsed}s")
        print(f"  Baloes: {detected} | Renderizados: {rendered} | "
              f"Detect: {timings.get('detect', 0):.2f}s | OCR: {timings.get('ocr', 0):.2f}s | "
              f"Traducao: {timings.get('translate', 0):.2f}s | Limpeza: {timings.get('clean', 0):.2f}s | "
              f"Render: {timings.get('render', 0):.2f}s")

    batch_elapsed = round(perf_counter() - batch_start, 2)
    avg_time = round(sum(page_times) / max(1, len(page_times)), 2)

    print(f"\n{'='*60}")
    print(f"[BATCH] Lote concluido")
    print(f"  Paginas processadas: {len(results)}/{total_files}")
    print(f"  Tempo total: {batch_elapsed}s")
    print(f"  Tempo medio por pagina: {avg_time}s")
    if page_times:
        slowest = max(page_times)
        fastest = min(page_times)
        print(f"  Pagina mais rapida: {fastest}s")
        print(f"  Pagina mais lenta: {slowest}s")
    print(f"{'='*60}\n")

    if progress_callback is not None:
        try:
            progress_callback(1.0, "Lote concluido")
        except TypeError:
            progress_callback(1.0, "Lote concluido")

    return results


def create_batch_zip(results: list[AppResult]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, result in enumerate(results):
            translated_path = Path(result.translated_image_path)
            if translated_path.exists():
                arcname = translated_path.name
                zf.write(translated_path, arcname)
    return buf.getvalue()
