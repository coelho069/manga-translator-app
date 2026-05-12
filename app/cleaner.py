from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.types import SpeechBubble


class BubbleCleaner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.last_inner_mask: np.ndarray | None = None
        self.last_ocr_mask: np.ndarray | None = None
        self.last_dark_mask: np.ndarray | None = None
        self.last_cleanup_mask: np.ndarray | None = None
        self.last_cleaned: bool = False

    def clean(self, image: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        self._reset_debug(image)

        if image is None or image.size == 0:
            return image

        inner_mask = self._inner_bubble_mask(
            bubble.mask,
            image.shape[:2],
            bubble,
        )
        self.last_inner_mask = inner_mask.copy()

        if cv2.countNonZero(inner_mask) == 0:
            self._add_note(bubble, "mascara interna vazia; limpeza ignorada")
            self.last_cleaned = False
            return image

        if self.config.use_dark_text_fallback:
            dark_mask = self._dark_text_mask(image, inner_mask, bubble)
        else:
            dark_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        ocr_mask = self._ocr_text_mask(bubble, image.shape[:2])

        self.last_dark_mask = dark_mask.copy()
        self.last_ocr_mask = ocr_mask.copy()

        final_mask = cv2.bitwise_or(dark_mask, ocr_mask)
        final_mask = cv2.bitwise_and(final_mask, inner_mask)
        final_mask = self._dilate(final_mask, self.config.text_mask_dilate_px)
        final_mask = self._close(final_mask, self.config.cleanup_morph_close_px)
        final_mask = self._dilate(final_mask, self.config.cleanup_extra_dilate_px)
        final_mask = cv2.bitwise_and(final_mask, inner_mask)

        final_mask = self._filter_if_too_large(final_mask, inner_mask, bubble)
        self.last_cleanup_mask = final_mask.copy()

        mask_pixels = cv2.countNonZero(final_mask)
        inner_pixels = cv2.countNonZero(inner_mask)

        if mask_pixels == 0:
            self._add_note(bubble, "mascara final de limpeza vazia; renderizacao ignorada")
            self.last_cleaned = False
            return image

        ratio = mask_pixels / max(1, inner_pixels)
        if ratio > self.config.max_cleanup_mask_ratio:
            self._add_note(
                bubble,
                f"mascara de limpeza grande demais ({ratio:.2f}); renderizacao ignorada",
            )
            self.last_cleaned = False
            return image

        if self.config.cleaner_mode == "inpaint":
            cleaned = cv2.inpaint(
                image,
                final_mask,
                self.config.inpaint_radius,
                cv2.INPAINT_TELEA,
            )
        else:
            cleaned = self._white_fill(image, final_mask, inner_mask)

        self.last_cleaned = True
        return cleaned

    def _reset_debug(self, image: np.ndarray | None) -> None:
        shape = image.shape[:2] if image is not None and image.size else (1, 1)
        empty = np.zeros(shape, dtype=np.uint8)
        self.last_inner_mask = empty.copy()
        self.last_ocr_mask = empty.copy()
        self.last_dark_mask = empty.copy()
        self.last_cleanup_mask = empty.copy()
        self.last_cleaned = False

    def _inner_bubble_mask(
        self,
        mask: np.ndarray | None,
        shape: tuple[int, int],
        bubble: SpeechBubble | None = None,
    ) -> np.ndarray:
        base = np.zeros(shape, dtype=np.uint8)

        if mask is not None and mask.size > 0:
            h = min(shape[0], mask.shape[0])
            w = min(shape[1], mask.shape[1])
            base[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255

        elif bubble is not None:
            margin = max(2, self.config.bubble_erode_px)
            x1 = max(0, bubble.bbox.x1 + margin)
            y1 = max(0, bubble.bbox.y1 + margin)
            x2 = min(shape[1], bubble.bbox.x2 - margin)
            y2 = min(shape[0], bubble.bbox.y2 - margin)
            if x2 > x1 and y2 > y1:
                base[y1:y2, x1:x2] = 255

        if cv2.countNonZero(base) == 0:
            return base

        radius = max(0, self.config.bubble_erode_px)
        while radius > 0:
            eroded = self._erode(base, radius)
            if cv2.countNonZero(eroded) > 0:
                return eroded
            radius //= 2

        return base

    def _dark_text_mask(
        self,
        image: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raw = np.zeros_like(gray, dtype=np.uint8)
        raw[gray < self.config.dark_text_threshold] = 255
        raw = cv2.bitwise_and(raw, inner_mask)

        filtered = self._filter_text_components(raw, inner_mask, bubble)
        filtered = self._close(filtered, self.config.cleanup_morph_close_px)

        return filtered

    def _ocr_text_mask(self, bubble: SpeechBubble, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)

        for box in getattr(bubble, "ocr_boxes", []):
            polygon = getattr(box, "polygon", None)
            if not polygon:
                continue

            points = np.asarray(polygon, dtype=np.int32)
            if len(points) < 3:
                continue

            cv2.fillPoly(mask, [points], 255)

        if cv2.countNonZero(mask) == 0:
            return mask

        mask = self._dilate(mask, max(1, self.config.text_mask_dilate_px // 2))
        mask = self._close(mask, self.config.cleanup_morph_close_px)

        return mask

    def _filter_text_components(
        self,
        mask: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)

        inner_area = max(1, cv2.countNonZero(inner_mask))
        max_component_area = max(
            self.config.min_text_component_area,
            int(inner_area * self.config.max_text_component_area_ratio),
        )

        bubble_w = max(1, bubble.bbox.x2 - bubble.bbox.x1)
        bubble_h = max(1, bubble.bbox.y2 - bubble.bbox.y1)

        for label_idx in range(1, num_labels):
            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            w = stats[label_idx, cv2.CC_STAT_WIDTH]
            h = stats[label_idx, cv2.CC_STAT_HEIGHT]
            area = stats[label_idx, cv2.CC_STAT_AREA]

            if area < self.config.min_text_component_area:
                continue

            if area > max_component_area:
                continue

            if w >= int(bubble_w * 0.85) or h >= int(bubble_h * 0.85):
                continue

            component_mask = (labels == label_idx).astype(np.uint8) * 255

            if self._component_touches_border_too_much(component_mask, inner_mask):
                continue

            filtered[labels == label_idx] = 255

        return filtered

    def _component_touches_border_too_much(
        self,
        component_mask: np.ndarray,
        inner_mask: np.ndarray,
    ) -> bool:
        border = cv2.subtract(inner_mask, self._erode(inner_mask, 1))
        if cv2.countNonZero(border) == 0:
            return False

        overlap = cv2.bitwise_and(component_mask, border)
        overlap_pixels = cv2.countNonZero(overlap)
        component_pixels = max(1, cv2.countNonZero(component_mask))

        return (overlap_pixels / component_pixels) > 0.35

    def _filter_if_too_large(
        self,
        final_mask: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        inner_pixels = cv2.countNonZero(inner_mask)
        if inner_pixels == 0:
            return np.zeros_like(final_mask)

        final_pixels = cv2.countNonZero(final_mask)
        ratio = final_pixels / inner_pixels

        if ratio <= self.config.max_cleanup_mask_ratio:
            return final_mask

        filtered = self._filter_text_components(final_mask, inner_mask, bubble)
        filtered = self._close(filtered, self.config.cleanup_morph_close_px)
        filtered = cv2.bitwise_and(filtered, inner_mask)

        filtered_pixels = cv2.countNonZero(filtered)
        filtered_ratio = filtered_pixels / inner_pixels

        if filtered_ratio <= self.config.max_cleanup_mask_ratio:
            return filtered

        return np.zeros_like(final_mask)

    def _white_fill(
        self,
        image: np.ndarray,
        final_mask: np.ndarray,
        inner_mask: np.ndarray,
    ) -> np.ndarray:
        result = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        light_pixels = (inner_mask > 0) & (gray > 200)

        if np.any(light_pixels):
            mean_color = image[light_pixels].mean(axis=0)
            fill_color = tuple(int(v) for v in mean_color)
        else:
            fill_color = (255, 255, 255)

        result[final_mask > 0] = fill_color
        return result

    @staticmethod
    def _kernel(radius: int) -> np.ndarray:
        size = max(1, radius * 2 + 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask
        return cv2.dilate(mask, self._kernel(radius), iterations=1)

    def _erode(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask
        return cv2.erode(mask, self._kernel(radius), iterations=1)

    def _close(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel(radius))

    @staticmethod
    def _add_note(bubble: SpeechBubble, message: str) -> None:
        if hasattr(bubble, "processing_notes") and isinstance(bubble.processing_notes, list):
            bubble.processing_notes.append(message)
