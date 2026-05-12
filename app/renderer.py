from __future__ import annotations

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont

from app.config import AppConfig
from app.types import SpeechBubble
from app.utils import cv2_to_pil, pil_to_cv2, safe_text


class TextRenderer:
    def __init__(self, config: AppConfig):
        self.config = config

    def render(self, image: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        text = safe_text(bubble.translated_text)
        if not text:
            return image
        if bubble.source_text and bubble.translated_text and not bubble.cleanup_success:
            bubble.processing_notes.append("limpeza nao confirmada; texto nao renderizado")
            return image

        rect = self.get_safe_text_rect(bubble, image.shape)
        if rect is None:
            bubble.processing_notes.append("area segura pequena demais; texto nao renderizado")
            return image

        x, y, w, h = rect
        padding = max(0, self.config.text_padding_px)
        x += padding
        y += padding
        w -= padding * 2
        h -= padding * 2
        if w < 12 or h < 12:
            bubble.processing_notes.append("area util pequena demais; texto nao renderizado")
            return image

        pil_image = cv2_to_pil(image)
        draw = ImageDraw.Draw(pil_image)
        font, lines, line_height, total_height = self._fit_text(draw, text, w, h, bubble)
        if font is None or not lines:
            bubble.processing_notes.append("texto grande demais; texto nao renderizado")
            return image

        if total_height > h or any(self._text_width(draw, line, font) > w for line in lines):
            bubble.processing_notes.append("texto final nao coube; texto nao renderizado")
            return image

        current_y = y + max(0, (h - total_height) // 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height_real = bbox[3] - bbox[1]
            line_width = bbox[2] - bbox[0]
            if current_y + line_height_real > y + h:
                bubble.processing_notes.append("linha ultrapassaria area segura; renderizacao interrompida")
                return image
            line_x = x + max(0, (w - line_width) // 2)
            if self.config.draw_text_outline:
                draw.text(
                    (line_x, current_y),
                    line,
                    fill=self.config.text_color,
                    font=font,
                    stroke_width=1,
                    stroke_fill=self.config.outline_color,
                )
            else:
                draw.text((line_x, current_y), line, fill=self.config.text_color, font=font)
            current_y += line_height

        return pil_to_cv2(pil_image)

    def get_safe_text_rect(self, bubble: SpeechBubble, image_shape) -> tuple[int, int, int, int] | None:
        image_h, image_w = image_shape[:2]
        min_w = max(12, self.config.min_bubble_width // 2)
        min_h = max(12, self.config.min_bubble_height // 2)

        if bubble.mask is not None and bubble.mask.size > 0:
            base_mask = np.zeros((image_h, image_w), dtype=np.uint8)
            h = min(image_h, bubble.mask.shape[0])
            w = min(image_w, bubble.mask.shape[1])
            base_mask[:h, :w] = (bubble.mask[:h, :w] > 0).astype(np.uint8) * 255

            erosion_candidates = [
                self.config.render_margin_px,
                max(1, self.config.render_margin_px // 2),
                max(0, self.config.render_margin_px // 3),
            ]
            for radius in erosion_candidates:
                safe_mask = self._erode_mask(base_mask, radius)
                rect = self._rect_from_mask(safe_mask)
                if rect is not None and rect[2] >= min_w and rect[3] >= min_h:
                    return rect

        margin_candidates = [
            self.config.render_margin_px,
            max(2, self.config.render_margin_px // 2),
            0,
        ]
        for margin in margin_candidates:
            x = max(0, bubble.bbox.x1 + margin)
            y = max(0, bubble.bbox.y1 + margin)
            x2 = min(image_w, bubble.bbox.x2 - margin)
            y2 = min(image_h, bubble.bbox.y2 - margin)
            w = max(0, x2 - x)
            h = max(0, y2 - y)
            if w >= min_w and h >= min_h:
                return x, y, w, h
        return None

    @staticmethod
    def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask.copy()
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(mask, kernel, iterations=1)

    @staticmethod
    def _rect_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None
        return cv2.boundingRect(coords)

    def _fit_text(self, draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int, bubble: SpeechBubble):
        step = max(1, self.config.font_shrink_step)
        effective_max_font = self._effective_max_font_size(bubble, max_width, max_height)
        for size in range(effective_max_font, self.config.min_font_size - 1, -step):
            font = self._load_font(size)
            for spacing_ratio in (self.config.line_spacing_ratio, 1.05, 1.0):
                lines = self._wrap_text(draw, text, font, max_width)
                if not lines:
                    continue
                line_height = self._line_height(draw, font, spacing_ratio)
                total_height = line_height * len(lines)
                widest = max(self._text_width(draw, line, font) for line in lines)
                if widest <= max_width and total_height <= max_height:
                    return font, lines, line_height, total_height

                narrowed_lines = [self._ellipsis(draw, line, font, max_width) for line in lines]
                narrowed_total_height = line_height * len(narrowed_lines)
                narrowed_widest = max(self._text_width(draw, line, font) for line in narrowed_lines)
                if narrowed_widest <= max_width and narrowed_total_height <= max_height:
                    return font, narrowed_lines, line_height, narrowed_total_height

                clipped_lines = self._clip_lines_to_height(draw, lines, font, line_height, max_width, max_height)
                if clipped_lines:
                    total_height = line_height * len(clipped_lines)
                    widest = max(self._text_width(draw, line, font) for line in clipped_lines)
                    if widest <= max_width and total_height <= max_height:
                        return font, clipped_lines, line_height, total_height
        return None, [], 0, 0

    def _effective_max_font_size(self, bubble: SpeechBubble, max_width: int, max_height: int) -> int:
        shortest_side = max(1, min(max_width, max_height, bubble.bbox.width, bubble.bbox.height))
        if shortest_side < 35:
            cap = 14
        elif shortest_side < 55:
            cap = 18
        elif shortest_side < 80:
            cap = 24
        else:
            cap = self.config.max_font_size
        return max(self.config.min_font_size, min(self.config.max_font_size, cap))

    @staticmethod
    def _load_font(size: int):
        for font_name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _wrap_text(self, draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
        words = safe_text(text).split()
        if not words:
            return []

        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if self._text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
            current = word

            while self._text_width(draw, current, font) > max_width and len(current) > 1:
                split_at = len(current)
                while split_at > 1 and self._text_width(draw, current[:split_at] + "-", font) > max_width:
                    split_at -= 1
                if split_at <= 1:
                    break
                lines.append(current[:split_at] + "-")
                current = current[split_at:]

        if current:
            lines.append(current)
        return lines

    def _clip_lines_to_height(
        self,
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        font,
        line_height: int,
        max_width: int,
        max_height: int,
    ) -> list[str]:
        if not lines or line_height <= 0:
            return []

        max_lines = max(1, max_height // max(1, line_height))
        if len(lines) <= max_lines:
            return lines

        clipped = lines[:max_lines]
        clipped[-1] = self._ellipsis(draw, clipped[-1], font, max_width)
        return clipped

    @staticmethod
    def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    @staticmethod
    def _line_height(draw: ImageDraw.ImageDraw, font, spacing_ratio: float) -> int:
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        return max(1, int((bbox[3] - bbox[1]) * max(1.0, spacing_ratio)))

    def _ellipsis(self, draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
        clean = safe_text(text)
        if self._text_width(draw, clean, font) <= max_width:
            return clean

        suffix = "..."
        while clean and self._text_width(draw, clean + suffix, font) > max_width:
            clean = clean[:-1]
        return safe_text(clean) + suffix if clean else suffix
