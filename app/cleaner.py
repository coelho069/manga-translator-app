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
        self.last_residual_dark_pixels: int = 0
        self.last_residual_ratio: float = 0.0
        self.last_block_reason: str = ""
        self.last_before_image: np.ndarray | None = None
        self.last_after_image: np.ndarray | None = None

    def clean(
        self,
        image: np.ndarray,
        bubble: SpeechBubble,
        exclusion_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        self._reset_debug(image)
        bubble.cleanup_success = False
        self.last_before_image = image.copy()

        if image is None or image.size == 0:
            self._add_note(bubble, "imagem invalida para limpeza")
            self.last_block_reason = "imagem invalida"
            return image

        print(f"[CLEANUP] start balloon={bubble.id} method=segmentation_mask")
        print(
            f"[CLEANUP] balloon_bbox=({bubble.bbox.x1},{bubble.bbox.y1},"
            f"{bubble.bbox.x2},{bubble.bbox.y2})"
        )
        inner_mask = self._inner_bubble_mask(
            mask=getattr(bubble, "mask", None),
            shape=image.shape[:2],
            bubble=bubble,
            exclusion_mask=exclusion_mask,
        )
        self.last_inner_mask = inner_mask.copy()

        ocr_mask = self._ocr_text_mask(bubble, image.shape[:2])
        ocr_coords = cv2.findNonZero(ocr_mask)
        if ocr_coords is not None:
            tx, ty, tw, th = cv2.boundingRect(ocr_coords)
            print(
                f"[CLEANUP] text_bbox=({tx},{ty},{tx + tw},{ty + th}) source=ocr_polygons"
            )
        else:
            print("[CLEANUP] text_bbox=none source=no_ocr_polygons")

        self.last_ocr_mask = ocr_mask.copy()
        self.last_dark_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Cleanup mask: area de texto detectada/fallback central, sempre
        # intersectada com a mascara YOLO do balao atual.
        cleanup_mask = self._build_cleanup_mask(
            bubble=bubble,
            image=image,
            shape=image.shape[:2],
            ocr_mask=ocr_mask,
            inner_mask=inner_mask,
            exclusion_mask=exclusion_mask,
        )

        cleanup_coords = cv2.findNonZero(cleanup_mask)
        cleanup_pixels = int(cv2.countNonZero(cleanup_mask))
        if cleanup_coords is not None:
            cx, cy, cw, ch = cv2.boundingRect(cleanup_coords)
            print(
                f"[CLEANUP] expanded_text_bbox=({cx},{cy},{cx + cw},{cy + ch})"
            )
            print(
                f"[CLEANUP_MASK] balloon={bubble.id} "
                f"bbox=({cx},{cy},{cx + cw},{cy + ch}) mask_area={cleanup_pixels}"
            )
        else:
            print("[CLEANUP] expanded_text_bbox=empty")
            print(f"[CLEANUP_MASK] balloon={bubble.id} bbox=empty mask_area=0")
        print(
            f"[CLEANUP_PLAN] balloon={bubble.id} "
            f"strategy=text_area_intersect_balloon_mask "
            f"has_bubble_mask={getattr(bubble, 'mask', None) is not None} "
            f"ocr_polygons={len(getattr(bubble, 'ocr_boxes', []))} "
            f"mask_area={cleanup_pixels}"
        )
        print(f"[CLEANUP] mask_area={cleanup_pixels}")

        self.last_cleanup_mask = cleanup_mask.copy()

        if cleanup_pixels == 0:
            self.last_cleaned = False
            bubble.cleanup_success = False
            self.last_block_reason = "cleanup_mask vazia"
            self._add_note(bubble, "falha na limpeza: cleanup_mask vazia")
            print(f"[CLEANUP_ERROR] balloon={bubble.id} error=cleanup_mask_empty")
            return image

        mode = safe_text(getattr(self.config, "cleaner_mode", "white_fill")).lower()
        print(f"[CLEANUP] method=segmentation_mask cleaner_mode={mode} balloon={bubble.id}")
        if mode == "inpaint":
            cleaned = cv2.inpaint(
                image,
                cleanup_mask,
                int(max(1, getattr(self.config, "inpaint_radius", 5))),
                cv2.INPAINT_TELEA,
            )
        else:
            cleaned = self._white_fill(image, cleanup_mask, cleanup_mask)

        self.last_after_image = cleaned.copy()

        verify_mask = cleanup_mask
        residual_ok, residual_count, residual_ratio = self._verify_cleanup(
            cleaned, verify_mask, bubble
        )

        if not residual_ok:
            print(
                f"[CLEANUP_FALLBACK] balloon={bubble.id} method=fill_inside_mask "
                f"reason=residual_{residual_ratio:.2%}_post_clean"
            )
            forced_mask = cleanup_mask
            forced_mask = self._apply_exclusion(forced_mask, exclusion_mask, image.shape[:2])
            self.last_cleanup_mask = forced_mask.copy()
            cleaned = self._white_fill(image, forced_mask, forced_mask)
            self.last_after_image = cleaned.copy()
            residual_ok, residual_count, residual_ratio = self._verify_cleanup(
                cleaned, verify_mask, bubble
            )

        if not residual_ok:
            self.last_cleaned = False
            bubble.cleanup_success = False
            self.last_block_reason = f"texto residual detectado: {residual_count} pixels ({residual_ratio:.1%})"
            self._add_note(bubble, f"limpeza falhou: texto residual {residual_count} pixels ({residual_ratio:.1%})")
            print(f"[CLEANER] Bubble {bubble.id}: BLOQUEADO - texto residual {residual_count} pixels ({residual_ratio:.1%})")
            return image

        self.last_cleaned = True
        bubble.cleanup_success = True
        self.last_block_reason = ""
        self.last_residual_dark_pixels = residual_count
        self.last_residual_ratio = residual_ratio
        print(f"[CLEANER] Bubble {bubble.id}: limpeza OK - residual {residual_count} pixels ({residual_ratio:.1%})")
        print(f"[CLEANUP_SUCCESS] balloon={bubble.id}")
        return cleaned

    def force_clean_bubble_inner_area(
        self,
        image: np.ndarray,
        bubble: SpeechBubble,
        exclusion_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        self._reset_debug(image)
        bubble.cleanup_success = False
        self.last_before_image = image.copy()

        if image is None or image.size == 0:
            self.last_block_reason = "imagem invalida"
            return image

        inner_mask = self._inner_bubble_mask(
            mask=getattr(bubble, "mask", None),
            shape=image.shape[:2],
            bubble=bubble,
            exclusion_mask=exclusion_mask,
        )
        if cv2.countNonZero(inner_mask) == 0:
            inner_mask = self._bbox_inner_mask(
                image.shape[:2], bubble, exclusion_mask=exclusion_mask
            )

        self.last_inner_mask = inner_mask.copy()
        self.last_ocr_mask = np.zeros_like(inner_mask)
        self.last_dark_mask = np.zeros_like(inner_mask)
        self.last_cleanup_mask = inner_mask.copy()

        if cv2.countNonZero(inner_mask) == 0:
            self.last_cleaned = False
            bubble.cleanup_success = False
            self.last_block_reason = "mascara interna vazia no fallback"
            return image

        print(
            f"[CLEANUP_FALLBACK] balloon={bubble.id} method=fill_inside_mask "
            f"reason={self.last_block_reason or 'forced_inner_fill'}"
        )
        cleaned = self._white_fill(image, inner_mask, inner_mask)
        self.last_after_image = cleaned.copy()

        residual_ok, residual_count, residual_ratio = self._verify_cleanup(
            cleaned, inner_mask, bubble
        )

        if not residual_ok:
            self.last_cleaned = False
            bubble.cleanup_success = False
            self.last_block_reason = f"fallback: texto residual {residual_count} pixels ({residual_ratio:.1%})"
            self._add_note(bubble, f"fallback falhou: texto residual {residual_count} pixels")
            return image

        self.last_cleaned = True
        bubble.cleanup_success = True
        self.last_block_reason = ""
        self.last_residual_dark_pixels = residual_count
        self.last_residual_ratio = residual_ratio
        self._add_note(bubble, "fallback forcado: limpeza branca interna aplicada")
        return cleaned

    def _verify_cleanup(
        self,
        cleaned_image: np.ndarray,
        inner_mask: np.ndarray,
        bubble: SpeechBubble,
    ) -> tuple[bool, int, float]:
        """Verify that the cleaned image has no significant residual text."""
        coords = cv2.findNonZero(inner_mask)
        if coords is None:
            return True, 0, 0.0

        x, y, w, h = cv2.boundingRect(coords)
        crop = cleaned_image[y : y + h, x : x + w]
        crop_inner = inner_mask[y : y + h, x : x + w]

        if crop.size == 0 or crop_inner.size == 0:
            return True, 0, 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        threshold = int(getattr(self.config, "residual_dark_threshold", 100))
        dark_pixels = np.zeros_like(gray, dtype=np.uint8)
        dark_pixels[gray < threshold] = 255
        dark_pixels = cv2.bitwise_and(dark_pixels, crop_inner)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark_pixels, connectivity=8
        )

        inner_area = max(1, cv2.countNonZero(crop_inner))
        residual_count = 0
        min_component_area = int(getattr(self.config, "residual_min_component_area", 6))
        max_component_ratio = float(getattr(self.config, "residual_max_component_ratio", 0.04))
        max_component_area = max(min_component_area, int(inner_area * max_component_ratio))

        bubble_w = max(1, bubble.bbox.width)
        bubble_h = max(1, bubble.bbox.height)
        max_w = int(bubble_w * 0.85)
        max_h = int(bubble_h * 0.85)

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            cw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            ch = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            if area < min_component_area:
                continue
            if area > max_component_area:
                continue
            if cw >= max_w or ch >= max_h:
                continue

            residual_count += area

        residual_ratio = residual_count / inner_area
        max_ratio = float(getattr(self.config, "residual_max_ratio", 0.025))

        return residual_ratio <= max_ratio, residual_count, residual_ratio

    def _reset_debug(self, image: np.ndarray | None) -> None:
        shape = image.shape[:2] if image is not None and image.size else (1, 1)
        empty = np.zeros(shape, dtype=np.uint8)
        self.last_inner_mask = empty.copy()
        self.last_ocr_mask = empty.copy()
        self.last_dark_mask = empty.copy()
        self.last_cleanup_mask = empty.copy()
        self.last_cleaned = False
        self.last_residual_dark_pixels = 0
        self.last_residual_ratio = 0.0
        self.last_block_reason = ""
        self.last_before_image = None
        self.last_after_image = None

    def _build_cleanup_mask(
        self,
        bubble: SpeechBubble,
        image: np.ndarray,
        shape: tuple[int, int],
        ocr_mask: np.ndarray,
        inner_mask: np.ndarray,
        exclusion_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Build text cleanup area and clamp it to the current YOLO bubble mask."""
        balloon_mask = np.zeros(shape, dtype=np.uint8)
        bubble_mask = getattr(bubble, "mask", None)
        if bubble_mask is not None and bubble_mask.size > 0:
            h = min(shape[0], bubble_mask.shape[0])
            w = min(shape[1], bubble_mask.shape[1])
            balloon_mask[:h, :w] = (bubble_mask[:h, :w] > 0).astype(np.uint8) * 255
        else:
            bx1 = max(0, min(shape[1], bubble.bbox.x1))
            by1 = max(0, min(shape[0], bubble.bbox.y1))
            bx2 = max(0, min(shape[1], bubble.bbox.x2))
            by2 = max(0, min(shape[0], bubble.bbox.y2))
            if bx2 > bx1 and by2 > by1:
                balloon_mask[by1:by2, bx1:bx2] = 255

        if cv2.countNonZero(balloon_mask) == 0:
            return balloon_mask

        text_area = np.zeros(shape, dtype=np.uint8)
        if ocr_mask is not None and cv2.countNonZero(ocr_mask) > 0:
            text_area = ocr_mask.copy()
        else:
            dark_mask = self._dark_text_mask(image, inner_mask, bubble)
            self.last_dark_mask = dark_mask.copy()
            if cv2.countNonZero(dark_mask) > 0:
                text_area = dark_mask
            else:
                text_area = self._central_cleanup_area(balloon_mask, bubble)

        text_area = self._refine_text_mask(text_area, balloon_mask)
        text_area = cv2.bitwise_and(text_area, balloon_mask)
        if exclusion_mask is None or exclusion_mask.size == 0:
            return text_area

        before = cv2.countNonZero(text_area)
        result = self._apply_exclusion(text_area, exclusion_mask, shape)
        after = cv2.countNonZero(result)
        if before > 0 and after < before:
            print(
                f"[CLEANUP_COLLISION] balloon={bubble.id} "
                f"other=neighbor pixels_excluded={before - after} action=reduce_margin"
            )
        return result

    def _central_cleanup_area(self, balloon_mask: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        coords = cv2.findNonZero(balloon_mask)
        fallback = np.zeros_like(balloon_mask)
        if coords is None:
            return fallback

        x, y, w, h = cv2.boundingRect(coords)
        margin_x = max(2, int(w * 0.14))
        margin_y = max(2, int(h * 0.14))
        x1 = max(x, x + margin_x)
        y1 = max(y, y + margin_y)
        x2 = min(x + w, x + w - margin_x)
        y2 = min(y + h, y + h - margin_y)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = x, y, x + w, y + h
        fallback[y1:y2, x1:x2] = 255
        fallback = cv2.bitwise_and(fallback, balloon_mask)
        self._add_note(bubble, "fallback central: limpeza limitada ao balao")
        return fallback

    @staticmethod
    def _apply_exclusion(
        mask: np.ndarray,
        exclusion_mask: np.ndarray | None,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if exclusion_mask is None or exclusion_mask.size == 0:
            return mask
        exclude = np.zeros(shape, dtype=np.uint8)
        eh = min(shape[0], exclusion_mask.shape[0])
        ew = min(shape[1], exclusion_mask.shape[1])
        exclude[:eh, :ew] = (exclusion_mask[:eh, :ew] > 0).astype(np.uint8) * 255
        return cv2.subtract(mask, exclude)

    def _inner_bubble_mask(
        self,
        mask: np.ndarray | None,
        shape: tuple[int, int],
        bubble: SpeechBubble,
        exclusion_mask: np.ndarray | None = None,
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

        if exclusion_mask is not None and exclusion_mask.size > 0:
            exclude = np.zeros(shape, dtype=np.uint8)
            eh = min(shape[0], exclusion_mask.shape[0])
            ew = min(shape[1], exclusion_mask.shape[1])
            exclude[:eh, :ew] = (exclusion_mask[:eh, :ew] > 0).astype(np.uint8) * 255
            before = cv2.countNonZero(base)
            base = cv2.subtract(base, exclude)
            after = cv2.countNonZero(base)
            if before > 0 and after < before:
                print(
                    f"[CLEANUP_COLLISION] balloon={bubble.id} "
                    f"pixels_excluded={before - after} action=mask_subtract"
                )

        if cv2.countNonZero(base) == 0:
            return base

        radius = max(0, int(getattr(self.config, "bubble_erode_px", 8)))
        while radius > 0:
            eroded = self._erode(base, radius)
            if cv2.countNonZero(eroded) > 0:
                return eroded
            radius //= 2

        return base

    def _bbox_inner_mask(
        self,
        shape: tuple[int, int],
        bubble: SpeechBubble,
        exclusion_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        margin = max(1, int(getattr(self.config, "bubble_erode_px", 8)))
        x1 = max(0, bubble.bbox.x1 + margin)
        y1 = max(0, bubble.bbox.y1 + margin)
        x2 = min(shape[1], bubble.bbox.x2 - margin)
        y2 = min(shape[0], bubble.bbox.y2 - margin)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

        if exclusion_mask is not None and exclusion_mask.size > 0:
            exclude = np.zeros(shape, dtype=np.uint8)
            eh = min(shape[0], exclusion_mask.shape[0])
            ew = min(shape[1], exclusion_mask.shape[1])
            exclude[:eh, :ew] = (exclusion_mask[:eh, :ew] > 0).astype(np.uint8) * 255
            mask = cv2.subtract(mask, exclude)
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

        dilate_radius = int(getattr(self.config, "text_mask_dilate_px", 15))
        close_radius = int(getattr(self.config, "cleanup_morph_close_px", 5))

        mask = self._dilate(mask, self._mode_radius(dilate_radius))
        mask = self._close(mask, self._mode_radius(close_radius))
        mask = self._dilate(mask, self._mode_radius(max(2, dilate_radius // 3)))
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

        dilate_radius = int(getattr(self.config, "text_mask_dilate_px", 15)) // 2
        close_radius = int(getattr(self.config, "cleanup_morph_close_px", 5))

        filtered_crop = self._dilate(filtered_crop, self._mode_radius(dilate_radius))
        filtered_crop = self._close(filtered_crop, self._mode_radius(close_radius))
        filtered_crop = self._dilate(filtered_crop, self._mode_radius(max(2, dilate_radius // 3)))

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
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

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
        mask = self._dilate(mask, self._mode_radius(3))
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

        # Texto cobre muita area do balao (>limit). Tentar filtrar componentes
        # arrisca deixar kanji grandes ou texto vertical de fora da limpeza.
        # Como ja temos a mascara de segmentacao do balao (com exclusao de
        # balões vizinhos via inner_mask), limpamos o interior inteiro - a
        # traducao sera renderizada por cima.
        print(
            f"[CLEANUP_FALLBACK] balloon={bubble.id} method=fill_inside_mask "
            f"reason=text_density_{ratio:.2f}_above_{limit:.2f}"
        )
        return inner_mask.copy()

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

        mask_coords = cv2.findNonZero(final_mask)
        if mask_coords is not None:
            mx, my, mw, mh = cv2.boundingRect(mask_coords)
            feather_radius = max(1, int(getattr(self.config, "cleanup_feather_px", 2)))
            for dy in range(-feather_radius, feather_radius + 1):
                for dx in range(-feather_radius, feather_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    shifted_mask = np.zeros_like(final_mask)
                    src_region = final_mask[
                        max(0, my + dy) : min(final_mask.shape[0], my + mh + dy),
                        max(0, mx + dx) : min(final_mask.shape[1], mx + mw + dx),
                    ]
                    dst_y = max(0, my + dy)
                    dst_x = max(0, mx + dx)
                    dst_h = min(final_mask.shape[0], my + mh + dy) - dst_y
                    dst_w = min(final_mask.shape[1], mx + mw + dx) - dst_x
                    if dst_h > 0 and dst_w > 0:
                        shifted_mask[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w] = src_region[:dst_h, :dst_w]

                    border = cv2.subtract(shifted_mask, final_mask)
                    border_inner = cv2.bitwise_and(border, inner_mask)
                    if cv2.countNonZero(border_inner) > 0:
                        result[border_inner > 0] = fill_color

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

    def _log_cleanup_status(
        self,
        bubble: SpeechBubble,
        ocr_pixels: int,
        dark_pixels: int,
        final_pixels: int,
    ) -> None:
        inner_pixels = cv2.countNonZero(self.last_inner_mask) if self.last_inner_mask is not None else 0
        print(f"[CLEANER] Bubble {bubble.id}: inner mask pixels = {inner_pixels}")
        print(f"[CLEANER] Bubble {bubble.id}: OCR mask pixels = {ocr_pixels}")
        print(f"[CLEANER] Bubble {bubble.id}: dark mask pixels = {dark_pixels}")
        print(f"[CLEANER] Bubble {bubble.id}: final mask pixels = {final_pixels}")
        print(f"[CLEANER] Bubble {bubble.id}: cleanup_success = {bubble.cleanup_success}")

    @staticmethod
    def _add_note(bubble: SpeechBubble, message: str) -> None:
        if hasattr(bubble, "processing_notes") and isinstance(bubble.processing_notes, list):
            bubble.processing_notes.append(safe_text(message))
