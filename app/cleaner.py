from __future__ import annotations

import cv2
import numpy as np

from app.config import AppConfig
from app.types import SpeechBubble
from app.utils import safe_text


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
        bubble.cleanup_success = False

        if image is None or image.size == 0:
            self._add_note(bubble, "imagem invalida para limpeza")
            return image

        inner_mask = self._inner_bubble_mask(
            mask=getattr(bubble, "mask", None),
            shape=image.shape[:2],
            bubble=bubble,
        )
        self.last_inner_mask = inner_mask.copy()

        if cv2.countNonZero(inner_mask) == 0:
            self._add_note(bubble, "mascara interna vazia")
            self.last_cleaned = False
            print(f"[CLEANER] Bubble {bubble.id}: OCR mask pixels = 0")
            print(f"[CLEANER] Bubble {bubble.id}: dark mask pixels = 0")
            print(f"[CLEANER] Bubble {bubble.id}: final mask pixels = 0")
            print(f"[CLEANER] Bubble {bubble.id}: cleanup_success = {bubble.cleanup_success}")
            return image

        ocr_mask = self._ocr_text_mask(bubble, image.shape[:2])
        dark_mask = (
            self._dark_text_mask(image=image, inner_mask=inner_mask, bubble=bubble)
            if bool(getattr(self.config, "use_dark_text_fallback", True))
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )

        self.last_ocr_mask = ocr_mask.copy()
        self.last_dark_mask = dark_mask.copy()

        ocr_pixels = cv2.countNonZero(ocr_mask)
        dark_pixels = cv2.countNonZero(dark_mask)

        final_mask = cv2.bitwise_or(ocr_mask, dark_mask)
        final_mask = cv2.bitwise_and(final_mask, inner_mask)

        if cv2.countNonZero(final_mask) > 0:
            final_mask = self._refine_text_mask(final_mask, inner_mask)
            final_mask = self._filter_if_too_large(final_mask, inner_mask, bubble)

        if cv2.countNonZero(final_mask) == 0:
            final_mask = self._fallback_inner_cleanup_mask(inner_mask, bubble)

        self.last_cleanup_mask = final_mask.copy()
        final_pixels = cv2.countNonZero(final_mask)

        print(f"[CLEANER] Bubble {bubble.id}: OCR mask pixels = {ocr_pixels}")
        print(f"[CLEANER] Bubble {bubble.id}: dark mask pixels = {dark_pixels}")
        print(f"[CLEANER] Bubble {bubble.id}: final mask pixels = {final_pixels}")

        if final_pixels == 0:
            self.last_cleaned = False
            bubble.cleanup_success = False
            self._add_note(bubble, "falha na limpeza: sem mascara final")
            print(f"[CLEANER] Bubble {bubble.id}: cleanup_success = {bubble.cleanup_success}")
            return image

        mode = safe_text(getattr(self.config, "cleaner_mode", "white_fill")).lower()
        if mode == "inpaint":
            cleaned = cv2.inpaint(
                image,
                final_mask,
                int(max(1, getattr(self.config, "inpaint_radius", 3))),
                cv2.INPAINT_TELEA,
            )
        else:
            cleaned = self._white_fill(image, final_mask, inner_mask)

        self.last_cleaned = True
        bubble.cleanup_success = True
        print(f"[CLEANER] Bubble {bubble.id}: cleanup_success = {bubble.cleanup_success}")
        return cleaned

    def force_clean_bubble_inner_area(self, image: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        self._reset_debug(image)
        bubble.cleanup_success = False

        if image is None or image.size == 0:
            return image

        inner_mask = self._inner_bubble_mask(
            mask=getattr(bubble, "mask", None),
            shape=image.shape[:2],
            bubble=bubble,
        )
        if cv2.countNonZero(inner_mask) == 0:
            inner_mask = self._bbox_inner_mask(image.shape[:2], bubble)

        self.last_inner_mask = inner_mask.copy()
        self.last_ocr_mask = np.zeros_like(inner_mask)
        self.last_dark_mask = np.zeros_like(inner_mask)
        self.last_cleanup_mask = inner_mask.copy()

        if cv2.countNonZero(inner_mask) == 0:
            self.last_cleaned = False
            bubble.cleanup_success = False
            return image

        cleaned = self._white_fill(image, inner_mask, inner_mask)
        self.last_cleaned = True
        bubble.cleanup_success = True
        self._add_note(bubble, "fallback forcado: limpeza branca interna aplicada")
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
        bubble: SpeechBubble,
    ) -> np.ndarray:
        base = np.zeros(shape, dtype=np.uint8)

        if mask is not None and mask.size > 0:
            h = min(shape[0], mask.shape[0])
            w = min(shape[1], mask.shape[1])
            base[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255
        else:
            margin = max(1, int(getattr(self.config, "bubble_erode_px", 8)))
            x1 = max(0, bubble.bbox.x1 + margin)
            y1 = max(0, bubble.bbox.y1 + margin)
            x2 = min(shape[1], bubble.bbox.x2 - margin)
            y2 = min(shape[0], bubble.bbox.y2 - margin)
            if x2 > x1 and y2 > y1:
                base[y1:y2, x1:x2] = 255

        if cv2.countNonZero(base) == 0:
            return base

        radius = max(0, int(getattr(self.config, "bubble_erode_px", 8)))
        while radius > 0:
            eroded = self._erode(base, radius)
            if cv2.countNonZero(eroded) > 0:
                return eroded
            radius //= 2

        return base

    def _bbox_inner_mask(self, shape: tuple[int, int], bubble: SpeechBubble) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        margin = max(1, int(getattr(self.config, "bubble_erode_px", 8)))
        x1 = max(0, bubble.bbox.x1 + margin)
        y1 = max(0, bubble.bbox.y1 + margin)
        x2 = min(shape[1], bubble.bbox.x2 - margin)
        y2 = min(shape[0], bubble.bbox.y2 - margin)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        return mask

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

        mask = self._dilate(mask, self._mode_radius(int(getattr(self.config, "text_mask_dilate_px", 15))))
        mask = self._close(mask, self._mode_radius(int(getattr(self.config, "cleanup_morph_close_px", 5))))
        return mask

    def _dark_text_mask(
        self,
        image: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        coords = cv2.findNonZero(inner_mask)
        if coords is None:
            return np.zeros(inner_mask.shape, dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)
        crop = image[y : y + h, x : x + w]
        crop_inner = inner_mask[y : y + h, x : x + w]
        if crop.size == 0 or crop_inner.size == 0:
            return np.zeros(inner_mask.shape, dtype=np.uint8)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        threshold = int(max(0, min(255, getattr(self.config, "dark_text_threshold", 200))))
        dark_crop = np.zeros_like(gray, dtype=np.uint8)
        dark_crop[gray < threshold] = 255
        dark_crop = cv2.bitwise_and(dark_crop, crop_inner)

        if cv2.countNonZero(dark_crop) == 0:
            return np.zeros(inner_mask.shape, dtype=np.uint8)

        filtered_crop = self._filter_text_components(dark_crop, crop_inner, bubble)
        if cv2.countNonZero(filtered_crop) == 0:
            return np.zeros(inner_mask.shape, dtype=np.uint8)

        filtered_crop = self._dilate(filtered_crop, self._mode_radius(int(getattr(self.config, "text_mask_dilate_px", 15)) // 2))
        filtered_crop = self._close(filtered_crop, self._mode_radius(int(getattr(self.config, "cleanup_morph_close_px", 5))))

        full = np.zeros(inner_mask.shape, dtype=np.uint8)
        full[y : y + h, x : x + w] = filtered_crop
        return full

    def _filter_text_components(
        self,
        mask: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)

        inner_area = max(1, cv2.countNonZero(inner_mask))
        min_area = int(max(1, getattr(self.config, "min_text_component_area", 4)))
        ratio_limit = float(max(0.01, getattr(self.config, "max_text_component_area_ratio", 0.12)))
        max_area = max(min_area, int(inner_area * ratio_limit))

        bubble_w = max(1, bubble.bbox.width)
        bubble_h = max(1, bubble.bbox.height)
        max_w = int(max(4, bubble_w * 0.90))
        max_h = int(max(4, bubble_h * 0.90))

        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])

            if area < min_area:
                continue
            if area > max_area:
                continue
            if w >= max_w or h >= max_h:
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

    def _refine_text_mask(self, text_mask: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
        mask = text_mask.copy()
        mask = self._dilate(mask, self._mode_radius(int(getattr(self.config, "cleanup_extra_dilate_px", 5))))
        mask = self._close(mask, self._mode_radius(int(getattr(self.config, "cleanup_morph_close_px", 5))))
        mask = cv2.bitwise_and(mask, inner_mask)
        return mask

    def _filter_if_too_large(
        self,
        final_mask: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        inner_pixels = cv2.countNonZero(inner_mask)
        if inner_pixels <= 0:
            return np.zeros_like(final_mask)

        limit = float(max(0.05, getattr(self.config, "max_cleanup_mask_ratio", 0.65)))
        final_pixels = cv2.countNonZero(final_mask)
        if final_pixels <= 0:
            return final_mask

        ratio = final_pixels / inner_pixels
        if ratio <= limit:
            return final_mask

        filtered = self._filter_text_components(final_mask, inner_mask, bubble)
        filtered = self._close(filtered, self._mode_radius(int(getattr(self.config, "cleanup_morph_close_px", 5))))
        filtered = cv2.bitwise_and(filtered, inner_mask)

        filtered_pixels = cv2.countNonZero(filtered)
        if filtered_pixels <= 0:
            return np.zeros_like(final_mask)

        filtered_ratio = filtered_pixels / inner_pixels
        if filtered_ratio <= limit:
            return filtered

        return np.zeros_like(final_mask)

    def _fallback_inner_cleanup_mask(self, inner_mask: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        allow_force = bool(getattr(self.config, "force_clean_on_failed_mask", True))
        allow_inner_fill = bool(getattr(self.config, "inner_white_fill_on_failed_cleanup", True))
        translated = safe_text(getattr(bubble, "translated_text", ""))

        if not allow_force or not allow_inner_fill or not translated:
            return np.zeros_like(inner_mask)

        self._add_note(bubble, "fallback agressivo: limpeza interna do balao")
        return inner_mask.copy()

    def _white_fill(
        self,
        image: np.ndarray,
        final_mask: np.ndarray,
        inner_mask: np.ndarray,
    ) -> np.ndarray:
        result = image.copy()
        coords = cv2.findNonZero(inner_mask)
        if coords is None:
            result[final_mask > 0] = (255, 255, 255)
            return result

        x, y, w, h = cv2.boundingRect(coords)
        image_crop = image[y : y + h, x : x + w]
        inner_crop = inner_mask[y : y + h, x : x + w]
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        bright = (inner_crop > 0) & (gray > 190)

        if np.any(bright):
            mean_color = image_crop[bright].mean(axis=0)
            fill_color = tuple(int(v) for v in mean_color)
        else:
            fill_color = (255, 255, 255)

        result[final_mask > 0] = fill_color
        return result

    def _mode_radius(self, radius: int) -> int:
        value = max(0, int(radius))
        if safe_text(getattr(self.config, "performance_mode", "balanced")).lower() == "fast":
            return max(0, value // 2)
        return value

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
            bubble.processing_notes.append(safe_text(message))
