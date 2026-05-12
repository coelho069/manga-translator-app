from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass
class OCRTextBox:
    text: str
    confidence: float
    polygon: list[tuple[int, int]]


@dataclass
class SpeechBubble:
    id: int
    bbox: BoundingBox
    mask: np.ndarray
    source_text: str = ""
    translated_text: str = ""
    ocr_boxes: list[OCRTextBox] = field(default_factory=list)
    processing_notes: list[str] = field(default_factory=list)
    cleanup_success: bool = False


@dataclass
class AppResult:
    original_image_path: Path
    translated_image_path: Path
    bubbles: list[SpeechBubble]
    output_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
