from __future__ import annotations

import hashlib
import threading

import numpy as np

from app.config import AppConfig
from app.manga_ocr_engine import configure_manga_ocr, run_manga_ocr
from app.types import OCRTextBox
from app.utils import resolve_ocr_lang, safe_text


class OCRReader:
    _readers = {}
    _cache = {}
    _lock = threading.Lock()
    _cache_lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.config = config
        self.ocr_lang = resolve_ocr_lang(config.ocr_lang or config.source_lang)
        self.ocr_engine = safe_text(config.ocr_engine).lower() or "auto"
        if self.ocr_engine not in {"auto", "paddle", "manga"}:
            self.ocr_engine = "auto"
        configure_manga_ocr(
            model_name=config.manga_ocr_model_name,
            cache_dir=config.manga_ocr_cache_dir,
            device=config.manga_ocr_device,
        )
        print(f"[OCR] idioma: {self.ocr_lang}")
        print(f"[OCR] motor: {self.ocr_engine}")

    def _get_reader(self):
        key = (self.ocr_lang, bool(self.config.use_gpu))
        with self._lock:
            if key not in OCRReader._readers:
                from paddleocr import PaddleOCR

                print(f"[MODEL] OCR carregado uma vez lang={self.ocr_lang}")
                try:
                    OCRReader._readers[key] = PaddleOCR(lang=self.ocr_lang, show_log=False, use_angle_cls=False)
                except TypeError:
                    OCRReader._readers[key] = PaddleOCR(lang=self.ocr_lang)
            return OCRReader._readers[key]

    def read(
        self,
        crop: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        force: bool = False,
        use_cache: bool = True,
    ) -> list[OCRTextBox]:
        if crop is None or crop.size == 0:
            print("[OCR] nenhum texto detectado")
            return []
        if not force and not self._has_text_signal(crop):
            print("[OCR] nenhum texto detectado")
            return []

        cache_key = self._crop_cache_key(crop)
        cached = self._get_cached(cache_key) if use_cache else None
        if use_cache and cached is not None and len(cached) > 0:
            print("[OCR] cache reutilizado")
            return self._offset_boxes(cached, offset_x, offset_y)

        if self._should_use_manga_ocr():
            boxes = self._read_manga(crop)
            if boxes:
                self._store_cached(cache_key, boxes)
                return self._offset_boxes(boxes, offset_x, offset_y)
            if self.ocr_engine == "manga":
                return []
            print("[MANGA-OCR] falha, usando fallback")

        reader = self._get_reader()
        try:
            raw_result = reader.ocr(crop, cls=False)
        except TypeError:
            raw_result = reader.ocr(crop)
        if not raw_result:
            print("[OCR] nenhum texto detectado")
            return []

        lines = raw_result[0] if len(raw_result) == 1 and isinstance(raw_result[0], list) else raw_result
        if not lines:
            print("[OCR] nenhum texto detectado")
            return []

        boxes: list[OCRTextBox] = []
        for item in lines:
            parsed = self._parse_item(item, 0, 0)
            if parsed is not None:
                boxes.append(parsed)
        if not boxes:
            print("[OCR] nenhum texto detectado")
        if boxes:
            self._store_cached(cache_key, boxes)
        return self._offset_boxes(boxes, offset_x, offset_y)

    def _should_use_manga_ocr(self) -> bool:
        is_japanese = self.ocr_lang in {"japan", "ja", "japanese"}
        return self.ocr_engine == "manga" or (self.ocr_engine == "auto" and is_japanese)

    def _read_manga(self, crop: np.ndarray) -> list[OCRTextBox]:
        text = run_manga_ocr(crop)
        text = safe_text(text)
        if not text:
            return []
        return [manga_ocr_text_to_ocr_box(text, crop.shape)]

    def _has_text_signal(self, crop: np.ndarray) -> bool:
        height, width = crop.shape[:2]
        if width < max(4, self.config.min_ocr_width // 2):
            return False
        if height < max(4, self.config.min_ocr_height // 2):
            return False

        if crop.ndim == 3:
            gray = crop.mean(axis=2)
        else:
            gray = crop
        dark_ratio = float(np.count_nonzero(gray < 230)) / max(1, gray.size)
        return dark_ratio >= self.config.min_ocr_dark_ratio

    def _crop_cache_key(self, crop: np.ndarray) -> tuple[str, str, bool, str]:
        contiguous = np.ascontiguousarray(crop)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.tobytes())
        return self.ocr_lang, self.ocr_engine, bool(self.config.use_gpu), digest.hexdigest()

    @classmethod
    def _get_cached(cls, key) -> list[OCRTextBox] | None:
        with cls._cache_lock:
            boxes = cls._cache.get(key)
            if boxes is None:
                return None
            return [
                OCRTextBox(
                    text=box.text,
                    confidence=box.confidence,
                    polygon=list(box.polygon),
                )
                for box in boxes
            ]

    def _store_cached(self, key, boxes: list[OCRTextBox]) -> None:
        max_items = max(16, int(self.config.ocr_cache_max_items))
        with OCRReader._cache_lock:
            OCRReader._cache[key] = [
                OCRTextBox(
                    text=box.text,
                    confidence=box.confidence,
                    polygon=list(box.polygon),
                )
                for box in boxes
            ]
            while len(OCRReader._cache) > max_items:
                OCRReader._cache.pop(next(iter(OCRReader._cache)))

    @staticmethod
    def _offset_boxes(boxes: list[OCRTextBox], offset_x: int, offset_y: int) -> list[OCRTextBox]:
        return [
            OCRTextBox(
                text=box.text,
                confidence=box.confidence,
                polygon=[(int(x) + offset_x, int(y) + offset_y) for x, y in box.polygon],
            )
            for box in boxes
        ]

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


def manga_ocr_text_to_ocr_box(text: str, crop_shape) -> OCRTextBox:
    height, width = crop_shape[:2]
    polygon = [(0, 0), (int(width), 0), (int(width), int(height)), (0, int(height))]
    return OCRTextBox(text=text, confidence=0.90, polygon=polygon)
