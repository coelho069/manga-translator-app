import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class AppConfig:
    bubble_model_path: Path = Path("models/bubble_seg.pt")
    hf_bubble_model_repo: str = "kitsumed/yolov8m_seg-speech-bubble"
    hf_bubble_model_filename: str = "model.pt"
    auto_download_bubble_model: bool = True
    output_dir: Path = Path("output")
    translation_flow: str = _env("DEFAULT_TRANSLATION_FLOW", "auto_to_en")
    translation_provider: str = _env("TRANSLATION_PROVIDER", "multilingual")
    hf_translation_model: str = _env("TRANSLATION_MODEL", _env("HF_TRANSLATION_MODEL", "facebook/m2m100_418M"))
    translation_refiner_enabled: bool = False
    source_lang: str = _env("DEFAULT_SOURCE_LANG", "auto")
    target_lang: str = _env("DEFAULT_TARGET_LANG", "en")
    ocr_lang: str = _env("OCR_LANG", "japan")
    ocr_engine: str = "auto"
    manga_ocr_model_name: str = "kha-white/manga-ocr-base"
    manga_ocr_cache_dir: str = "models/huggingface"
    manga_ocr_device: str = "cpu"
    translation_style: str = "natural"
    performance_mode: str = "balanced"
    yolo_confidence: float = 0.20
    yolo_iou: float = 0.50
    yolo_imgsz: int = 960
    yolo_max_det: int = 60
    bubble_erode_px: int = 4
    text_mask_dilate_px: int = 10
    render_margin_px: int = 10
    translation_font_scale: float = 1.0
    min_font_size: int = 9
    max_font_size: int = 32
    inpaint_radius: int = 3
    use_gpu: bool = False
    min_bubble_area: int = 300
    min_bubble_width: int = 20
    min_bubble_height: int = 20
    min_ocr_area: int = 900
    min_ocr_width: int = 24
    min_ocr_height: int = 18
    min_ocr_dark_ratio: float = 0.001
    ocr_cache_max_items: int = 256
    bubble_mask_close_px: int = 5
    bubble_mask_open_px: int = 3
    use_dark_text_fallback: bool = True
    dark_text_threshold: int = 200
    max_cleanup_mask_ratio: float = 0.65
    min_text_component_area: int = 4
    max_text_component_area_ratio: float = 0.12
    cleanup_morph_close_px: int = 7
    cleanup_extra_dilate_px: int = 3
    cleaner_mode: str = "white_fill"
    force_clean_on_failed_mask: bool = True
    inner_white_fill_on_failed_cleanup: bool = True
    line_spacing_ratio: float = 1.12
    font_shrink_step: int = 1
    max_render_font_attempts: int = 28
    text_padding_px: int = 6
    draw_text_outline: bool = False
    text_color: tuple[int, int, int] = (0, 0, 0)
    outline_color: tuple[int, int, int] = (255, 255, 255)
    debug_enabled: bool = False
    debug_dir: Path | None = None
