from __future__ import annotations

import threading

import numpy as np

from app.config import AppConfig
from app.types import OCRTextBox
from app.utils import safe_text


class OCRReader:
    _reader = None
    _lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.config = config

    def _get_reader(self):
        with self._lock:
            if OCRReader._reader is None:
                from paddleocr import PaddleOCR

                OCRReader._reader = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=self.config.use_gpu,
                    show_log=False,
                )
            return OCRReader._reader

    def read(self, crop: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> list[OCRTextBox]:
        if crop is None or crop.size == 0:
            return []

        reader = self._get_reader()
        raw_result = reader.ocr(crop, cls=True)
        if not raw_result:
            return []

        lines = raw_result[0] if len(raw_result) == 1 and isinstance(raw_result[0], list) else raw_result
        if not lines:
            return []

        boxes: list[OCRTextBox] = []
        for item in lines:
            parsed = self._parse_item(item, offset_x, offset_y)
            if parsed is not None:
                boxes.append(parsed)
        return boxes

    @staticmethod
    def _parse_item(item, offset_x: int, offset_y: int) -> OCRTextBox | None:
        try:
            polygon_raw = item[0]
            text_info = item[1]
            text = safe_text(text_info[0] if isinstance(text_info, (list, tuple)) else text_info)
            confidence = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 0.0
            if not text:
                return None

            polygon: list[tuple[int, int]] = []
            for point in polygon_raw:
                x = int(round(float(point[0]))) + offset_x
                y = int(round(float(point[1]))) + offset_y
                polygon.append((x, y))

            if len(polygon) < 3:
                return None
            return OCRTextBox(text=text, confidence=confidence, polygon=polygon)
        except (TypeError, ValueError, IndexError):
            return None

