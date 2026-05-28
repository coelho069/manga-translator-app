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

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x1 + self.width // 2, self.y1 + self.height // 2

    def clamp(self, image_width: int, image_height: int) -> "BoundingBox":
        return BoundingBox(
            x1=max(0, min(self.x1, image_width - 1)),
            y1=max(0, min(self.y1, image_height - 1)),
            x2=max(0, min(self.x2, image_width)),
            y2=max(0, min(self.y2, image_height)),
        )


@dataclass
class OCRTextBox:
    text: str
    confidence: float
    polygon: list[tuple[int, int]]


@dataclass
class SpeechBubble:
    id: int
    bbox: BoundingBox
    mask: np.ndarray | None = None
    source_text: str = ""
    translated_text: str = ""
    ocr_boxes: list[OCRTextBox] = field(default_factory=list)
    processing_notes: list[str] = field(default_factory=list)
    cleanup_success: bool = False
    render_success: bool = False


@dataclass
class AppResult:
    original_image_path: Path
    bubbles: list
    translated_image_path: Path | None = None
    output_dir: Path | None = None
    image_bytes: bytes | None = None
    metadata: dict = field(default_factory=dict)
