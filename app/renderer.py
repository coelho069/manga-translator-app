from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont

from app.config import AppConfig
from app.types import SpeechBubble
from app.utils import cv2_to_pil, pil_to_cv2, safe_text, write_image_cv2


class TextRenderer:
    """
    Renderiza o texto traduzido dentro do balao com:
    - area segura centralizada
    - uso da mascara do balao quando disponivel
    - quebra de linha inteligente
    - reducao automatica da fonte
    - centralizacao horizontal e vertical
    - fallback com reticencias quando o texto ainda for grande demais
    - verificacao de mascara de limpeza antes de renderizar
    - contencao total do texto dentro do balao
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.debug_dir: Path | None = None
        self.last_font_size: int = 0
        self.last_line_count: int = 0
        self.last_text_width: int = 0
        self.last_text_height: int = 0
        self.last_shrink_reason: str = ""

    def render(self, image: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        text = safe_text(getattr(bubble, "translated_text", ""))
        if not text:
            return image

        if safe_text(getattr(bubble, "source_text", "")) and not bool(getattr(bubble, "cleanup_success", False)):
            if hasattr(bubble, "processing_notes") and isinstance(bubble.processing_notes, list):
                bubble.processing_notes.append("renderizacao ignorada: limpeza nao confirmada")
            return image

        rect = self.get_safe_text_rect(bubble, image.shape)
        if rect is None:
            bubble.processing_notes.append("area segura pequena demais; texto nao renderizado")
            return image

        x, y, w, h = rect

        padding = max(0, int(getattr(self.config, "text_padding_px", 8)))
        x += padding
        y += padding
        w -= padding * 2
        h -= padding * 2

        if w < 12 or h < 12:
            bubble.processing_notes.append("area util pequena demais; texto nao renderizado")
            return image

        pil_image = cv2_to_pil(image)
        draw = ImageDraw.Draw(pil_image)

        font, lines, line_height, total_height = self._fit_text(
            draw=draw,
            text=text,
            max_width=w,
            max_height=h,
            bubble=bubble,
        )

        if font is None or not lines:
            bubble.processing_notes.append("texto grande demais; texto nao renderizado")
            return image

        if total_height > h:
            bubble.processing_notes.append("altura final do texto excedeu o balao")
            return image

        if any(self._text_width(draw, line, font) > w for line in lines):
            bubble.processing_notes.append("largura final do texto excedeu o balao")
            return image

        center_text = bool(getattr(self.config, "center_text", True))
        if center_text:
            current_y = y + max(0, (h - total_height) // 2)
        else:
            current_y = y

        widest = max(self._text_width(draw, line, font) for line in lines)

        self.last_font_size = getattr(font, "size", 0) if hasattr(font, "size") else 0
        self.last_line_count = len(lines)
        self.last_text_width = widest
        self.last_text_height = total_height

        print(f"[RENDERER] Bubble {bubble.id}: font_size={self.last_font_size}px, "
              f"lines={self.last_line_count}, "
              f"text={self.last_text_width}x{self.last_text_height}px, "
              f"area={w}x{h}px")
        if self.last_shrink_reason:
            print(f"[RENDERER] Bubble {bubble.id}: motivo reducao: {self.last_shrink_reason}")

        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_height_real = line_bbox[3] - line_bbox[1]

            if current_y + line_height_real > y + h:
                bubble.processing_notes.append("linha ultrapassaria area segura; renderizacao interrompida")
                return image

            if center_text:
                line_x = x + max(0, (w - line_width) // 2)
            else:
                line_x = x

            if bool(getattr(self.config, "draw_text_outline", False)):
                draw.text(
                    (line_x, current_y),
                    line,
                    fill=getattr(self.config, "text_color", (0, 0, 0)),
                    font=font,
                    stroke_width=1,
                    stroke_fill=getattr(self.config, "outline_color", (255, 255, 255)),
                )
            else:
                draw.text(
                    (line_x, current_y),
                    line,
                    fill=getattr(self.config, "text_color", (0, 0, 0)),
                    font=font,
                )

            current_y += line_height

        result = pil_to_cv2(pil_image)

        if getattr(bubble, "mask", None) is not None and bubble.mask.size > 0:
            result = self._contain_render_inside_mask(result, image, bubble)

        self._save_debug_images(bubble, image, result, x, y, w, h, lines, font, draw)

        return result

    def _save_debug_images(
        self,
        bubble: SpeechBubble,
        original: np.ndarray,
        rendered: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        lines: list[str],
        font,
        draw: ImageDraw.ImageDraw,
    ) -> None:
        if self.debug_dir is None:
            return

        debug_layout = rendered.copy()
        cv2.rectangle(debug_layout, (x, y), (x + w, y + h), (0, 200, 0), 1)
        cv2.rectangle(debug_layout, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)

        text_bottom = y
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            lh = line_bbox[3] - line_bbox[1]
            text_bottom += lh
        cv2.line(debug_layout, (x, text_bottom), (x + w, text_bottom), (255, 0, 0), 1)

        write_image_cv2(Path(self.debug_dir) / "debug_text_layout.png", debug_layout)

        debug_font = rendered.copy()
        font_info = f"font:{self.last_font_size}px lines:{self.last_line_count} size:{self.last_text_width}x{self.last_text_height}"
        cv2.putText(
            debug_font,
            font_info,
            (max(5, x), max(15, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        write_image_cv2(Path(self.debug_dir) / "debug_font_size.png", debug_font)

        debug_box = original.copy()
        cv2.rectangle(debug_box, (bubble.bbox.x1, bubble.bbox.y1), (bubble.bbox.x2, bubble.bbox.y2), (0, 180, 0), 2)
        cv2.rectangle(debug_box, (x, y), (x + w, y + h), (0, 0, 255), 1)
        write_image_cv2(Path(self.debug_dir) / "debug_text_box.png", debug_box)

    def _contain_render_inside_mask(
        self,
        rendered: np.ndarray,
        original: np.ndarray,
        bubble: SpeechBubble,
    ) -> np.ndarray:
        """Ensure rendered text stays inside the bubble mask."""
        mask = bubble.mask
        if mask is None or mask.size == 0:
            return rendered

        h, w = rendered.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if mask_h != h or mask_w != w:
            return rendered

        inner_mask = (mask > 0).astype(np.uint8)
        erode_radius = max(1, int(getattr(self.config, "bubble_erode_px", 8)) // 2)
        kernel_size = erode_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inner_mask = cv2.erode(inner_mask, kernel, iterations=1)

        outside_mask = (inner_mask == 0).astype(np.uint8)

        if cv2.countNonZero(outside_mask) == 0:
            return rendered

        result = rendered.copy()
        result[outside_mask > 0] = original[outside_mask > 0]
        return result

    def get_safe_text_rect(self, bubble: SpeechBubble, image_shape) -> tuple[int, int, int, int] | None:
        """
        Retorna uma area central segura para o texto.

        Importante:
        Para baloes ovais ou irregulares, usar apenas o bounding box pode deixar
        o texto sair visualmente do balao nos cantos. Por isso, quando existe
        mascara, esta funcao tenta encontrar um retangulo central com alta
        cobertura dentro da mascara.
        """

        image_h, image_w = image_shape[:2]

        min_w = max(12, int(getattr(self.config, "min_bubble_width", 20)) // 2)
        min_h = max(12, int(getattr(self.config, "min_bubble_height", 20)) // 2)

        if getattr(bubble, "mask", None) is not None and bubble.mask.size > 0:
            rect = self._safe_rect_from_mask(bubble.mask, image_w, image_h, min_w, min_h)
            if rect is not None:
                return rect

        return self._safe_rect_from_bbox(bubble, image_w, image_h, min_w, min_h)

    def _safe_rect_from_mask(
        self,
        mask: np.ndarray,
        image_w: int,
        image_h: int,
        min_w: int,
        min_h: int,
    ) -> tuple[int, int, int, int] | None:
        base_mask = np.zeros((image_h, image_w), dtype=np.uint8)

        h = min(image_h, mask.shape[0])
        w = min(image_w, mask.shape[1])
        base_mask[:h, :w] = (mask[:h, :w] > 0).astype(np.uint8) * 255

        if cv2.countNonZero(base_mask) == 0:
            return None

        margin = max(0, int(getattr(self.config, "render_margin_px", 10)))
        erosion_candidates = [
            margin,
            max(1, margin // 2),
            max(0, margin // 3),
            0,
        ]

        best_rect: tuple[int, int, int, int] | None = None
        best_area = 0

        for radius in erosion_candidates:
            safe_mask = self._erode_mask(base_mask, radius)
            if cv2.countNonZero(safe_mask) == 0:
                continue

            bbox = self._rect_from_mask(safe_mask)
            if bbox is None:
                continue

            candidate = self._central_rect_inside_mask(safe_mask, bbox, min_w, min_h)
            if candidate is None:
                continue

            area = candidate[2] * candidate[3]
            if area > best_area:
                best_rect = candidate
                best_area = area

        return best_rect

    def _central_rect_inside_mask(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        min_w: int,
        min_h: int,
    ) -> tuple[int, int, int, int] | None:
        x, y, w, h = bbox
        if w < min_w or h < min_h:
            return None

        crop = mask[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        dist = cv2.distanceTransform((crop > 0).astype(np.uint8), cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        center_x = x + int(max_loc[0])
        center_y = y + int(max_loc[1])

        scale_candidates = [
            (0.78, 0.64),
            (0.72, 0.70),
            (0.86, 0.56),
            (0.66, 0.76),
            (0.58, 0.82),
            (0.90, 0.48),
            (0.50, 0.88),
        ]

        best_rect: tuple[int, int, int, int] | None = None
        best_area = 0

        for sx, sy in scale_candidates:
            candidate_w = max(min_w, int(w * sx))
            candidate_h = max(min_h, int(h * sy))

            rect = self._rect_centered_and_clamped(
                center_x=center_x,
                center_y=center_y,
                width=candidate_w,
                height=candidate_h,
                bounds=(x, y, w, h),
            )

            rect = self._shrink_until_inside_mask(mask, rect, min_w, min_h)
            if rect is None:
                continue

            area = rect[2] * rect[3]
            if area > best_area:
                best_rect = rect
                best_area = area

        return best_rect

    @staticmethod
    def _rect_centered_and_clamped(
        center_x: int,
        center_y: int,
        width: int,
        height: int,
        bounds: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        bx, by, bw, bh = bounds
        x = int(center_x - width // 2)
        y = int(center_y - height // 2)

        x = max(bx, min(x, bx + bw - width))
        y = max(by, min(y, by + bh - height))

        return x, y, width, height

    def _shrink_until_inside_mask(
        self,
        mask: np.ndarray,
        rect: tuple[int, int, int, int],
        min_w: int,
        min_h: int,
    ) -> tuple[int, int, int, int] | None:
        x, y, w, h = rect

        while w >= min_w and h >= min_h:
            roi = mask[y : y + h, x : x + w]
            if roi.size == 0:
                return None

            coverage = cv2.countNonZero(roi) / float(w * h)

            if coverage >= 0.96:
                return x, y, w, h

            shrink_x = max(1, int(w * 0.04))
            shrink_y = max(1, int(h * 0.04))
            x += shrink_x
            y += shrink_y
            w -= shrink_x * 2
            h -= shrink_y * 2

        return None

    def _safe_rect_from_bbox(
        self,
        bubble: SpeechBubble,
        image_w: int,
        image_h: int,
        min_w: int,
        min_h: int,
    ) -> tuple[int, int, int, int] | None:
        margin = max(0, int(getattr(self.config, "render_margin_px", 10)))

        margin_candidates = [
            margin,
            max(2, margin // 2),
            0,
        ]

        for current_margin in margin_candidates:
            x = max(0, bubble.bbox.x1 + current_margin)
            y = max(0, bubble.bbox.y1 + current_margin)
            x2 = min(image_w, bubble.bbox.x2 - current_margin)
            y2 = min(image_h, bubble.bbox.y2 - current_margin)

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

    def _font_scale(self) -> float:
        try:
            scale = float(getattr(self.config, "translation_font_scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        if scale <= 0:
            return 1.0
        return min(3.0, max(0.25, scale))

    def _scaled_font_bounds(self) -> tuple[int, int]:
        base_min = max(1, int(getattr(self.config, "min_font_size", 9)))
        base_max = max(base_min, int(getattr(self.config, "max_font_size", 32)))
        scale = self._font_scale()

        scaled_min = max(1, int(round(base_min * scale)))
        scaled_max = max(scaled_min, int(round(base_max * scale)))
        return scaled_min, scaled_max

    def _fit_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        max_height: int,
        bubble: SpeechBubble,
    ):
        min_font_size, _ = self._scaled_font_bounds()
        step = max(1, int(getattr(self.config, "font_shrink_step", 1)))
        effective_max_font = self._effective_max_font_size(bubble, max_width, max_height)
        auto_resize = bool(getattr(self.config, "auto_font_resize", True))

        spacing_candidates = self._spacing_candidates()
        max_attempts = int(getattr(self.config, "max_render_font_attempts", 0) or 0)
        attempts = 0
        self.last_shrink_reason = ""

        if not auto_resize:
            font = self._load_font(effective_max_font)
            target_width = max_width
            lines = self._wrap_text(draw, text, font, target_width)
            if lines:
                lines = self._balance_lines(draw, lines, font, target_width)
                spacing_ratio = spacing_candidates[0] if spacing_candidates else 1.12
                line_height = self._line_height(draw, font, spacing_ratio)
                total_height = line_height * len(lines)
                widest = max(self._text_width(draw, line, font) for line in lines)

                if widest <= max_width and total_height <= max_height:
                    return font, lines, line_height, total_height

                self.last_shrink_reason = "auto_font_resize=False, texto nao cabe"
                return None, [], 0, 0

            self.last_shrink_reason = "auto_font_resize=False, wrap falhou"
            return None, [], 0, 0

        for size in range(effective_max_font, min_font_size - 1, -step):
            attempts += 1
            if max_attempts and attempts > max_attempts:
                self.last_shrink_reason = f"max tentativas ({max_attempts}) atingido"
                break

            font = self._load_font(size)

            target_width_candidates = (
                max_width,
                int(max_width * 0.92),
                int(max_width * 0.84),
            )

            for target_width in target_width_candidates:
                if target_width < 8:
                    continue

                lines = self._wrap_text(draw, text, font, target_width)
                if not lines:
                    continue

                lines = self._balance_lines(draw, lines, font, target_width)

                for spacing_ratio in spacing_candidates:
                    line_height = self._line_height(draw, font, spacing_ratio)
                    total_height = line_height * len(lines)
                    widest = max(self._text_width(draw, line, font) for line in lines)

                    if widest <= max_width and total_height <= max_height:
                        if size < effective_max_font:
                            self.last_shrink_reason = f"reduzido de {effective_max_font} para {size}px"
                        return font, lines, line_height, total_height

                    clipped_lines = self._clip_lines_to_height(
                        draw=draw,
                        lines=lines,
                        font=font,
                        line_height=line_height,
                        max_width=max_width,
                        max_height=max_height,
                    )

                    if clipped_lines:
                        total_height = line_height * len(clipped_lines)
                        widest = max(self._text_width(draw, line, font) for line in clipped_lines)

                        if widest <= max_width and total_height <= max_height:
                            if size < effective_max_font:
                                self.last_shrink_reason = f"reduzido de {effective_max_font} para {size}px + clip"
                            return font, clipped_lines, line_height, total_height

        self.last_shrink_reason = f"nenhum tamanho coube (min={min_font_size}px, max={effective_max_font}px)"
        return None, [], 0, 0

    def _effective_max_font_size(self, bubble: SpeechBubble, max_width: int, max_height: int) -> int:
        scale = self._font_scale()
        scaled_min, scaled_max = self._scaled_font_bounds()
        base_config_max = int(getattr(self.config, "max_font_size", 32))

        shortest_side = max(1, min(max_width, max_height, bubble.bbox.width, bubble.bbox.height))

        if shortest_side < 32:
            cap = 12
        elif shortest_side < 45:
            cap = 15
        elif shortest_side < 65:
            cap = 20
        elif shortest_side < 90:
            cap = 25
        else:
            cap = base_config_max

        scaled_cap = max(1, int(round(cap * scale)))
        return max(scaled_min, min(scaled_max, scaled_cap))

    def _load_font(self, size: int):
        bold = bool(getattr(self.config, "bold_text", False))
        return self._load_font_cached(size, bold)

    @staticmethod
    @lru_cache(maxsize=192)
    def _load_font_cached(size: int, bold: bool = False):
        if bold:
            font_candidates = (
                "arialbd.ttf",
                "Arial Bold.ttf",
                "Arial-Bold.ttf",
                "DejaVuSans-Bold.ttf",
                "LiberationSans-Bold.ttf",
                "arial.ttf",
                "Arial.ttf",
                "DejaVuSans.ttf",
                "LiberationSans-Regular.ttf",
            )
        else:
            font_candidates = (
                "arial.ttf",
                "Arial.ttf",
                "DejaVuSans.ttf",
                "DejaVuSans-Bold.ttf",
                "LiberationSans-Regular.ttf",
            )

        for font_name in font_candidates:
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _spacing_candidates(self) -> tuple[float, ...]:
        base = float(getattr(self.config, "line_spacing_ratio", 1.12))
        performance_mode = safe_text(getattr(self.config, "performance_mode", "balanced")).lower()

        if performance_mode == "fast":
            return (base, 1.0)

        if performance_mode == "quality":
            return (base, 1.10, 1.05, 1.0, 0.96)

        return (base, 1.05, 1.0, 0.96)

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

            if self._text_width(draw, word, font) <= max_width:
                current = word
            else:
                broken, remainder = self._break_long_word(draw, word, font, max_width)
                lines.extend(broken)
                current = remainder

        if current:
            lines.append(current)

        return [line for line in lines if safe_text(line)]

    def _break_long_word(
        self,
        draw: ImageDraw.ImageDraw,
        word: str,
        font,
        max_width: int,
    ) -> tuple[list[str], str]:
        remaining = safe_text(word)
        lines: list[str] = []

        while remaining and self._text_width(draw, remaining, font) > max_width:
            split_at = len(remaining)

            while split_at > 1 and self._text_width(draw, remaining[:split_at] + "-", font) > max_width:
                split_at -= 1

            if split_at <= 1:
                return lines, remaining

            lines.append(remaining[:split_at] + "-")
            remaining = remaining[split_at:]

        return lines, remaining

    def _balance_lines(
        self,
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        font,
        max_width: int,
    ) -> list[str]:
        if len(lines) < 2:
            return lines

        balanced = lines[:]

        changed = True
        while changed:
            changed = False

            for i in range(len(balanced) - 1):
                current_words = balanced[i].split()
                next_line = balanced[i + 1]

                if len(current_words) <= 1:
                    continue

                last_word = current_words[-1]
                candidate_current = " ".join(current_words[:-1])
                candidate_next = f"{last_word} {next_line}"

                if self._text_width(draw, candidate_next, font) > max_width:
                    continue

                current_width = self._text_width(draw, balanced[i], font)
                next_width = self._text_width(draw, next_line, font)
                new_current_width = self._text_width(draw, candidate_current, font)
                new_next_width = self._text_width(draw, candidate_next, font)

                old_diff = abs(current_width - next_width)
                new_diff = abs(new_current_width - new_next_width)

                if new_diff < old_diff:
                    balanced[i] = candidate_current
                    balanced[i + 1] = candidate_next
                    changed = True

        return balanced

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
        raw_height = bbox[3] - bbox[1]
        return max(1, int(raw_height * max(0.90, spacing_ratio)))

    def _ellipsis(self, draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
        clean = safe_text(text)
        if not clean:
            return ""

        if self._text_width(draw, clean, font) <= max_width:
            return clean

        suffix = "..."
        while clean and self._text_width(draw, clean + suffix, font) > max_width:
            clean = clean[:-1]

        return safe_text(clean) + suffix if clean else suffix
