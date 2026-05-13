from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.types import BoundingBox, SpeechBubble
from app.utils import ensure_dir, write_image_cv2


class BubbleDetector:
    _model_cache = {}

    def __init__(self, config: AppConfig):
        self.config = config
        self.model_path = self._ensure_model_file()

        from ultralytics import YOLO

        cache_key = (str(self.model_path.resolve()), bool(config.use_gpu), "segment")
        if cache_key not in BubbleDetector._model_cache:
            BubbleDetector._model_cache[cache_key] = YOLO(str(self.model_path), task="segment")
            print("[YOLO] modelo carregado")
        self.model = BubbleDetector._model_cache[cache_key]

    def detect(self, image: np.ndarray) -> list[SpeechBubble]:
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        results = self.model.predict(
            image,
            task="segment",
            imgsz=self.config.yolo_imgsz,
            conf=self.config.yolo_confidence,
            iou=self.config.yolo_iou,
            device="cpu" if not self.config.use_gpu else None,
            verbose=False,
        )

        bubbles: list[SpeechBubble] = []
        if not results:
            self._save_detection_debug(image, bubbles)
            print("[YOLO] baloes detectados: 0")
            return bubbles

        result = results[0]
        confidences = self._result_confidences(result)
        box_fallbacks = self._result_boxes(result, width, height)
        mask_polygons = self._result_mask_polygons(result)
        detection_count = max(len(mask_polygons), len(box_fallbacks))

        for idx in range(detection_count):
            mask = np.zeros((height, width), dtype=np.uint8)
            has_mask = False

            polygon = mask_polygons[idx] if idx < len(mask_polygons) else None
            if polygon is not None and len(polygon) >= 3:
                points = np.asarray(polygon, dtype=np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                cv2.fillPoly(mask, [points], 255)
                has_mask = cv2.countNonZero(mask) > 0

            if not has_mask and idx < len(box_fallbacks):
                mask = self._mask_from_bbox(box_fallbacks[idx], (height, width))

            processed_mask = self._post_process_mask(mask) if has_mask else mask
            if cv2.countNonZero(processed_mask) == 0:
                processed_mask = mask
            bbox = self._bbox_from_mask(processed_mask)
            if bbox is None:
                continue

            bubble = SpeechBubble(id=0, bbox=bbox, mask=processed_mask)
            if idx < len(confidences):
                bubble.processing_notes.append(f"confidence={confidences[idx]:.3f}")
            if not has_mask:
                bubble.processing_notes.append("fallback_bbox_sem_mascara")
            bubbles.append(bubble)

        bubbles.sort(key=lambda item: (item.bbox.y1, item.bbox.x1))
        for index, bubble in enumerate(bubbles, start=1):
            bubble.id = index

        self._save_detection_debug(image, bubbles)
        print(f"[YOLO] baloes detectados: {len(bubbles)}")
        return bubbles

    def _ensure_model_file(self) -> Path:
        model_path = Path(self.config.bubble_model_path)
        if model_path.exists():
            return model_path

        if not self.config.auto_download_bubble_model:
            raise FileNotFoundError(
                f"Modelo YOLO nao encontrado em '{model_path}'. "
                "Coloque o arquivo bubble_seg.pt dentro da pasta models/."
            )

        print(f"[HF] baixando modelo {self.config.hf_bubble_model_repo}...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "A dependencia huggingface_hub nao esta instalada. "
                "Execute: python -m pip install -r requirements.txt"
            ) from exc

        cached_path = hf_hub_download(
            repo_id=self.config.hf_bubble_model_repo,
            filename=self.config.hf_bubble_model_filename,
        )
        ensure_dir(model_path.parent)
        shutil.copy2(cached_path, model_path)
        print(f"[HF] modelo salvo em {model_path}")
        return model_path

    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        processed = (mask > 0).astype(np.uint8) * 255
        processed = self._morph(processed, cv2.MORPH_CLOSE, self.config.bubble_mask_close_px)
        processed = self._morph(processed, cv2.MORPH_OPEN, self.config.bubble_mask_open_px)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(processed)

        cleaned = np.zeros_like(processed)
        min_contour_area = max(12, self.config.min_bubble_area // 10)
        for contour in contours:
            if cv2.contourArea(contour) >= min_contour_area:
                cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)

        if cv2.countNonZero(cleaned) == 0:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)

        if self.config.performance_mode == "fast":
            return cleaned.astype(np.uint8)

        blur = cv2.GaussianBlur(cleaned, (3, 3), 0)
        _, smoothed = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return smoothed.astype(np.uint8)

    @staticmethod
    def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
        if radius <= 0 or cv2.countNonZero(mask) == 0:
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

    @staticmethod
    def _result_confidences(result) -> list[float]:
        try:
            if result.boxes is None or result.boxes.conf is None:
                return []
            return [float(value) for value in result.boxes.conf.detach().cpu().numpy().tolist()]
        except Exception:
            return []

    @staticmethod
    def _result_mask_polygons(result) -> list:
        try:
            if result.masks is None or result.masks.xy is None:
                return []
            return list(result.masks.xy)
        except Exception:
            return []

    @staticmethod
    def _result_boxes(result, image_w: int, image_h: int) -> list[BoundingBox]:
        try:
            if result.boxes is None or result.boxes.xyxy is None:
                return []
            raw_boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
        except Exception:
            return []

        boxes: list[BoundingBox] = []
        for raw in raw_boxes:
            if len(raw) < 4:
                continue
            x1 = max(0, min(image_w - 1, int(round(float(raw[0])))))
            y1 = max(0, min(image_h - 1, int(round(float(raw[1])))))
            x2 = max(0, min(image_w, int(round(float(raw[2])))))
            y2 = max(0, min(image_h, int(round(float(raw[3])))))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        return boxes

    @staticmethod
    def _mask_from_bbox(bbox: BoundingBox, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] = 255
        return mask

    def _save_detection_debug(self, image: np.ndarray, bubbles: list[SpeechBubble]) -> None:
        output_dir = ensure_dir(self.config.output_dir)
        debug_detection = image.copy()
        debug_masks = image.copy()
        boxes = []

        overlay = np.zeros_like(image)
        for bubble in bubbles:
            color = self._color_for_id(bubble.id)
            mask_bool = bubble.mask > 0
            overlay[mask_bool] = color
            cv2.rectangle(
                debug_detection,
                (bubble.bbox.x1, bubble.bbox.y1),
                (bubble.bbox.x2, bubble.bbox.y2),
                color,
                2,
            )
            cv2.putText(
                debug_detection,
                str(bubble.id),
                (bubble.bbox.x1, max(0, bubble.bbox.y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            boxes.append(
                {
                    "id": bubble.id,
                    "bbox": {
                        "x1": bubble.bbox.x1,
                        "y1": bubble.bbox.y1,
                        "x2": bubble.bbox.x2,
                        "y2": bubble.bbox.y2,
                    },
                    "area": int(cv2.countNonZero(bubble.mask)),
                    "confidence": self._confidence_from_notes(bubble.processing_notes),
                    "has_mask": "fallback_bbox_sem_mascara" not in bubble.processing_notes,
                    "notes": list(bubble.processing_notes),
                }
            )

        write_image_cv2(output_dir / "debug_detection.png", debug_detection)

        if bubbles:
            debug_masks = cv2.addWeighted(debug_masks, 0.72, overlay, 0.28, 0)

        write_image_cv2(output_dir / "debug_masks.png", debug_masks)
        (output_dir / "debug_boxes.json").write_text(
            json.dumps(boxes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _confidence_from_notes(notes: list[str]) -> float | None:
        for note in notes:
            if not note.startswith("confidence="):
                continue
            try:
                return float(note.split("=", 1)[1])
            except (IndexError, ValueError):
                return None
        return None

    @staticmethod
    def _color_for_id(value: int) -> tuple[int, int, int]:
        palette = [
            (32, 214, 163),
            (124, 92, 255),
            (255, 184, 77),
            (90, 169, 255),
            (255, 107, 138),
        ]
        return palette[(value - 1) % len(palette)]
