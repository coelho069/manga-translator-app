from __future__ import annotations

import threading

import numpy as np

from app.config import AppConfig
from app.types import OCRTextBox
from app.utils import resolve_ocr_lang, safe_text


class OCRReader:
    _readers = {}
    _lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.config = config
        self.ocr_lang = resolve_ocr_lang(config.ocr_lang or config.source_lang)

    def _get_reader(self):
        key = (self.ocr_lang, bool(self.config.use_gpu))
        with self._lock:
            if key not in OCRReader._readers:
                from paddleocr import PaddleOCR

                OCRReader._readers[key] = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.ocr_lang,
                    use_gpu=self.config.use_gpu,
                    show_log=False,
                )
            return OCRReader._readers[key]

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
