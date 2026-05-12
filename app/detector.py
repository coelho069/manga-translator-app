from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.types import BoundingBox, SpeechBubble


class BubbleDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model_path = Path(config.bubble_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo YOLO nao encontrado em '{self.model_path}'. "
                "Coloque o arquivo bubble_seg.pt dentro da pasta models/."
            )

        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> list[SpeechBubble]:
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        results = self.model.predict(
            image,
            conf=self.config.yolo_confidence,
            iou=self.config.yolo_iou,
            device="cpu" if not self.config.use_gpu else None,
            verbose=False,
        )

        bubbles: list[SpeechBubble] = []
        if not results:
            return bubbles

        result = results[0]
        if result.masks is None or result.masks.xy is None:
            return bubbles

        for polygon in result.masks.xy:
            if polygon is None or len(polygon) < 3:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.asarray(polygon, dtype=np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [points], 255)

            processed_mask = self._post_process_mask(mask)
            bbox = self._bbox_from_mask(processed_mask)
            if bbox is None or not self._is_valid_bubble(processed_mask, bbox):
                continue

            bubbles.append(SpeechBubble(id=0, bbox=bbox, mask=processed_mask))

        bubbles.sort(key=lambda item: (item.bbox.y1, item.bbox.x1))
        for index, bubble in enumerate(bubbles, start=1):
            bubble.id = index
        return bubbles

    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        processed = (mask > 0).astype(np.uint8) * 255
        processed = self._morph(processed, cv2.MORPH_CLOSE, self.config.bubble_mask_close_px)
        processed = self._morph(processed, cv2.MORPH_OPEN, self.config.bubble_mask_open_px)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(processed)

        largest = max(contours, key=cv2.contourArea)
        cleaned = np.zeros_like(processed)
        cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)

        blur = cv2.GaussianBlur(cleaned, (3, 3), 0)
        _, smoothed = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return smoothed.astype(np.uint8)

    @staticmethod
    def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, operation, kernel, iterations=1)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> BoundingBox | None:
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)

    def _is_valid_bubble(self, mask: np.ndarray, bbox: BoundingBox) -> bool:
        area = int(cv2.countNonZero(mask))
        if area < self.config.min_bubble_area:
            return False
        if bbox.width < self.config.min_bubble_width:
            return False
        if bbox.height < self.config.min_bubble_height:
            return False
        return True
