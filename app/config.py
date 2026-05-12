from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    bubble_model_path: Path = Path("models/bubble_seg.pt")
    output_dir: Path = Path("output")
    source_lang: str = "en"
    target_lang: str = "pt"
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    bubble_erode_px: int = 8
    text_mask_dilate_px: int = 11
    render_margin_px: int = 10
    min_font_size: int = 10
    max_font_size: int = 42
    inpaint_radius: int = 3
    use_gpu: bool = False
    min_bubble_area: int = 300
    min_bubble_width: int = 20
    min_bubble_height: int = 20
    bubble_mask_close_px: int = 5
    bubble_mask_open_px: int = 3
    use_dark_text_fallback: bool = True
    dark_text_threshold: int = 190
    max_cleanup_mask_ratio: float = 0.65
    min_text_component_area: int = 4
    max_text_component_area_ratio: float = 0.12
    cleanup_morph_close_px: int = 5
    cleanup_extra_dilate_px: int = 3
    cleaner_mode: str = "white_fill"
    line_spacing_ratio: float = 1.12
    font_shrink_step: int = 1
    text_padding_px: int = 4
    draw_text_outline: bool = False
    text_color: tuple[int, int, int] = (0, 0, 0)
    outline_color: tuple[int, int, int] = (255, 255, 255)
    debug_dir: Path | None = None
