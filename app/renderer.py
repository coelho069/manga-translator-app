from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import AppConfig
from app.types import SpeechBubble
from app.utils import cv2_to_pil, pil_to_cv2, safe_text


class TextRenderer:
    """
    Renderiza o texto traduzido dentro do balão com:
    - área segura centralizada
    - uso da máscara do balão quando disponível
    - quebra de linha inteligente
    - redução automática da fonte
    - centralização horizontal e vertical
    - fallback com reticências quando o texto ainda for grande demais
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._occupied_mask: np.ndarray | None = None
        self._bubble_context: list[SpeechBubble] = []

    def set_bubble_context(self, bubbles: list[SpeechBubble]) -> None:
        self._bubble_context = list(bubbles or [])

    def reset_occupancy(self, image_shape) -> None:
        self._occupied_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    def render(self, image: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        bubble.render_success = False
        text = safe_text(getattr(bubble, "translated_text", ""))
        bbox_text = f"{bubble.bbox.x1},{bubble.bbox.y1},{bubble.bbox.x2},{bubble.bbox.y2}"
        print(f"[RENDER_INPUT] balloon={bubble.id} text=\"{text}\" bbox={bbox_text}")
        if not text:
            self._render_fail(bubble, "texto traduzido vazio")
            return image
        print(f"[RENDERER] desenhando balao {bubble.id}")
        print(f"[RENDERER] desenhando traducao no balao {bubble.id}")
        print(f"[RENDERER] texto={text}")
        if self._occupied_mask is None or self._occupied_mask.shape != image.shape[:2]:
            self.reset_occupancy(image.shape)

        if bubble.bbox.width <= 5 or bubble.bbox.height <= 5:
            return self._fallback_render(image, bubble, "bbox invalido ou pequeno demais")

        rect = self.get_safe_text_rect(bubble, image.shape)
        if rect is None:
            return self._fallback_render(image, bubble, "area segura pequena demais; texto nao renderizado")

        x, y, w, h = rect

        padding = max(0, min(8, int(getattr(self.config, "text_padding_px", 8))))
        x += padding
        y += padding
        w -= padding * 2
        h -= padding * 2

        if w <= 5 or h <= 5:
            return self._fallback_render(image, bubble, "area util pequena demais; texto nao renderizado")

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
            return self._fallback_render(image, bubble, "texto grande demais ou fonte compativel ausente; texto nao renderizado")

        font_size = max(8, int(getattr(font, "size", 8) or 8))
        print(f"[RENDER_VALIDATE] balloon={bubble.id} width={w} height={h} font_size={font_size} lines={len(lines)}")

        if total_height > h:
            return self._fallback_render(image, bubble, "altura final do texto excedeu o balao")

        if any(self._text_width(draw, line, font) > w for line in lines):
            return self._fallback_render(image, bubble, "largura final do texto excedeu o balao")

        current_y = y + max(0, (h - total_height) // 2)
        planned_lines: list[tuple[int, int, str]] = []
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            line_height_real = line_bbox[3] - line_bbox[1]

            if current_y + line_height_real > y + h:
                return self._fallback_render(image, bubble, "linha ultrapassaria area segura; renderizacao interrompida")

            line_x = x + max(0, (w - line_width) // 2)

            planned_lines.append((line_x, current_y, line))
            current_y += line_height

        text_mask = self._text_mask_for_lines(image.shape, planned_lines, font)
        safe_mask = self._safe_mask_for_bubble(bubble, image.shape)
        if not self._text_mask_inside_safe_area(text_mask, safe_mask):
            return self._fallback_render(image, bubble, "texto ultrapassaria area segura; renderizacao bloqueada")
        if self._occupied_mask is not None and cv2.countNonZero(cv2.bitwise_and(text_mask, self._occupied_mask)) > 0:
            return self._fallback_render(image, bubble, "texto colide com outra traducao; renderizacao bloqueada")

        print(f"[RENDERER] font_size={getattr(font, 'size', 'default')}")
        print(f"[RENDER_DRAW] balloon={bubble.id} x={x} y={y} font_size={font_size} lines={len(planned_lines)}")
        for line_x, current_y, line in planned_lines:
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

        if self._occupied_mask is not None:
            self._occupied_mask[text_mask > 0] = 255
        bubble.render_success = True
        print("[RENDERER] sucesso")
        print(f"[RENDER_SUCCESS] balloon={bubble.id}")

        return pil_to_cv2(pil_image)

    @staticmethod
    def _render_fail(bubble: SpeechBubble, reason: str) -> None:
        if hasattr(bubble, "processing_notes") and isinstance(bubble.processing_notes, list):
            bubble.processing_notes.append(reason)
        print(f"[RENDERER] falhou balao {getattr(bubble, 'id', '?')}: {reason}")
        print(f"[RENDER_FAIL] balloon={getattr(bubble, 'id', '?')} reason={reason}")

    def _fallback_render(self, image: np.ndarray, bubble: SpeechBubble, reason: str) -> np.ndarray:
        print(f"[RENDER_FALLBACK] balloon={bubble.id} reason={reason}")
        text = safe_text(getattr(bubble, "translated_text", ""))
        if not text or image is None or image.size == 0:
            self._render_fail(bubble, reason)
            return image

        image_h, image_w = image.shape[:2]
        margin = 4
        x = max(0, min(image_w - 1, int(bubble.bbox.x1) + margin))
        y = max(0, min(image_h - 1, int(bubble.bbox.y1) + margin))
        x2 = max(x + 1, min(image_w, int(bubble.bbox.x2) - margin))
        y2 = max(y + 1, min(image_h, int(bubble.bbox.y2) - margin))
        w = x2 - x
        h = y2 - y
        if w <= 5 or h <= 5:
            self._render_fail(bubble, "fallback sem bbox valido")
            return image

        pil_image = cv2_to_pil(image)
        draw = ImageDraw.Draw(pil_image)
        font_size = max(8, min(24, int(h * 0.22), int(w * 0.16)))
        font = None
        lines: list[str] = []
        line_height = 0
        total_height = 0

        for size in range(font_size, 7, -1):
            font = self._load_font(
                size=size,
                target_lang=safe_text(getattr(self.config, "target_lang", "")),
                sample_text=text,
                preferred_font=safe_text(getattr(self.config, "font_path", "")),
            ) or ImageFont.load_default()
            lines = self._wrap_text(draw, text, font, max(6, w))
            if not lines:
                lines = [self._ellipsis(draw, text, font, max(6, w))]
            line_height = self._line_height(draw, font, 1.0)
            max_lines = max(1, h // max(1, line_height))
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = self._ellipsis(draw, lines[-1], font, max(6, w))
            total_height = line_height * len(lines)
            if lines and total_height <= h and all(self._text_width(draw, line, font) <= w for line in lines):
                font_size = size
                break

        if font is None or not lines:
            self._render_fail(bubble, "fallback nao conseguiu fonte/linhas")
            return image

        print(f"[RENDER_VALIDATE] balloon={bubble.id} width={w} height={h} font_size={font_size} lines={len(lines)}")
        current_y = max(y, min(y2 - 1, y + max(0, (h - total_height) // 2)))
        drawn_lines = 0
        for line in lines:
            line = self._ellipsis(draw, line, font, w)
            if not line:
                continue
            line_width = self._text_width(draw, line, font)
            line_x = max(x, min(x2 - 1, x + max(0, (w - line_width) // 2)))
            if current_y >= y2:
                break
            draw.text((line_x, current_y), line, fill=(0, 0, 0), font=font)
            drawn_lines += 1
            current_y += line_height

        if drawn_lines <= 0:
            self._render_fail(bubble, "fallback nao desenhou linhas")
            return image

        print(f"[RENDER_DRAW] balloon={bubble.id} x={x} y={y} font_size={font_size} lines={drawn_lines}")
        bubble.render_success = True
        print(f"[RENDER_SUCCESS] balloon={bubble.id}")
        return pil_to_cv2(pil_image)

    def get_safe_text_rect(self, bubble: SpeechBubble, image_shape) -> tuple[int, int, int, int] | None:
        """
        Retorna uma área central segura para o texto.

        Importante:
        Para balões ovais ou irregulares, usar apenas o bounding box pode deixar
        o texto sair visualmente do balão nos cantos. Por isso, quando existe
        máscara, esta função tenta encontrar um retângulo central com alta
        cobertura dentro da máscara.
        """

        image_h, image_w = image_shape[:2]

        min_w = max(12, int(getattr(self.config, "min_bubble_width", 20)) // 2)
        min_h = max(12, int(getattr(self.config, "min_bubble_height", 20)) // 2)

        if getattr(bubble, "mask", None) is not None and bubble.mask.size > 0:
            rect = self._safe_rect_from_mask(bubble.mask, image_w, image_h, min_w, min_h, bubble)
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
        bubble: SpeechBubble,
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
            safe_mask = self._exclude_other_bubbles(safe_mask, bubble)
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

        # Escalas conservadoras. Em balões ovais, uma área central menor
        # é mais confiável do que usar o bbox inteiro.
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

            # 0.96 evita usar cantos fora de balões ovais/irregulares.
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

    def _safe_mask_for_bubble(self, bubble: SpeechBubble, image_shape) -> np.ndarray:
        image_h, image_w = image_shape[:2]
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        raw = getattr(bubble, "mask", None)
        margin = max(0, int(getattr(self.config, "render_margin_px", 10)))
        if raw is not None and raw.size > 0:
            h = min(image_h, raw.shape[0])
            w = min(image_w, raw.shape[1])
            mask[:h, :w] = (raw[:h, :w] > 0).astype(np.uint8) * 255
            eroded = self._erode_mask(mask, margin)
            mask = eroded if cv2.countNonZero(eroded) > 0 else mask
        else:
            x1 = max(0, bubble.bbox.x1 + margin)
            y1 = max(0, bubble.bbox.y1 + margin)
            x2 = min(image_w, bubble.bbox.x2 - margin)
            y2 = min(image_h, bubble.bbox.y2 - margin)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        return self._exclude_other_bubbles(mask, bubble)

    def _exclude_other_bubbles(self, mask: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        if mask is None or mask.size == 0 or not self._bubble_context:
            return mask
        safe = mask.copy()
        shape = safe.shape[:2]
        other_mask = np.zeros(shape, dtype=np.uint8)
        for other in self._bubble_context:
            if other is bubble or getattr(other, "id", None) == getattr(bubble, "id", None):
                continue
            raw = getattr(other, "mask", None)
            if raw is not None and raw.size > 0:
                h = min(shape[0], raw.shape[0])
                w = min(shape[1], raw.shape[1])
                other_mask[:h, :w] = cv2.bitwise_or(other_mask[:h, :w], ((raw[:h, :w] > 0).astype(np.uint8) * 255))
            else:
                margin = max(1, int(getattr(self.config, "bubble_erode_px", 8)) // 2)
                x1 = max(0, other.bbox.x1 + margin)
                y1 = max(0, other.bbox.y1 + margin)
                x2 = min(shape[1], other.bbox.x2 - margin)
                y2 = min(shape[0], other.bbox.y2 - margin)
                if x2 > x1 and y2 > y1:
                    other_mask[y1:y2, x1:x2] = 255
        if cv2.countNonZero(other_mask) == 0:
            return safe
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        other_mask = cv2.dilate(other_mask, kernel, iterations=1)
        safe[other_mask > 0] = 0
        return safe

    @staticmethod
    def _text_mask_inside_safe_area(text_mask: np.ndarray, safe_mask: np.ndarray) -> bool:
        if cv2.countNonZero(text_mask) == 0 or cv2.countNonZero(safe_mask) == 0:
            return False
        outside = text_mask.copy()
        outside[safe_mask > 0] = 0
        return cv2.countNonZero(outside) == 0

    def _text_mask_for_lines(self, image_shape, line_positions: list[tuple[int, int, str]], font) -> np.ndarray:
        height, width = image_shape[:2]
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        stroke_width = 1 if bool(getattr(self.config, "draw_text_outline", False)) else 0
        for x, y, line in line_positions:
            draw.text((x, y), line, fill=255, font=font, stroke_width=stroke_width, stroke_fill=255)
        return np.array(mask_image, dtype=np.uint8)

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

        spacing_candidates = self._spacing_candidates()
        max_attempts = int(getattr(self.config, "max_render_font_attempts", 0) or 0)
        attempts = 0

        for size in range(effective_max_font, min_font_size - 1, -step):
            attempts += 1
            if max_attempts and attempts > max_attempts:
                break

            font = self._load_font(
                size=size,
                target_lang=safe_text(getattr(self.config, "target_lang", "")),
                sample_text=text,
                preferred_font=safe_text(getattr(self.config, "font_path", "")),
            )
            if font is None:
                continue

            # Testa múltiplas larguras-alvo para melhorar a centralização visual.
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
                            return font, clipped_lines, line_height, total_height

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

    @classmethod
    @lru_cache(maxsize=256)
    def _load_font(cls, size: int, target_lang: str = "", sample_text: str = "", preferred_font: str = ""):
        font_path = cls.resolve_font_for_language(target_lang, preferred_font, sample_text)
        if font_path:
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                print(f"[FONT] falha ao carregar fonte: {font_path}")

        if cls._contains_cjk(sample_text) or cls._is_cjk_language(target_lang):
            print("[FONT] nenhuma fonte CJK compativel encontrada; texto CJK nao deve ser renderizado com fonte padrao")
            return None

        for font_name in cls._latin_font_candidates():
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    @classmethod
    def resolve_font_for_language(cls, target_lang: str, preferred_font: str = "", sample_text: str = "") -> str | None:
        target = safe_text(target_lang).lower()
        text = safe_text(sample_text)

        candidates: list[str] = []
        if preferred_font:
            candidates.append(preferred_font)

        if cls._is_japanese_language(target) or cls._contains_japanese(text):
            candidates.extend(cls._japanese_font_candidates())
        elif cls._is_chinese_language(target) or cls._contains_chinese(text):
            candidates.extend(cls._chinese_font_candidates())
        elif cls._contains_cjk(text):
            candidates.extend(cls._cjk_font_candidates())
        else:
            candidates.extend(cls._latin_font_candidates())
            candidates.extend(cls._cjk_font_candidates())

        for candidate in cls._unique_candidates(candidates):
            if not candidate:
                continue
            if not cls._font_file_exists(candidate):
                continue
            if cls.font_supports_text(candidate, text):
                if preferred_font and candidate != preferred_font and (cls._contains_cjk(text) or cls._is_cjk_language(target)):
                    print("[FONT] fallback CJK aplicado")
                print(f"[FONT] fonte escolhida: {candidate}")
                return candidate
            if preferred_font and candidate == preferred_font:
                print("[FONT] fonte nao suporta texto, procurando outra")

        return None

    @staticmethod
    def font_supports_text(font_path: str, text: str) -> bool:
        sample = safe_text(text)
        if not sample:
            return True
        try:
            font = ImageFont.truetype(font_path, size=24)
        except OSError:
            return False

        missing_glyph_masks = set()
        for missing_char in ("\ufffd", "\u25a1", "\u25a0"):
            try:
                missing_mask = font.getmask(missing_char)
                missing_bbox = missing_mask.getbbox()
                if missing_bbox is not None:
                    missing_glyph_masks.add((missing_bbox, bytes(missing_mask)))
            except Exception:
                continue

        unsupported = 0
        checked = 0
        for char in sample:
            if char.isspace():
                continue
            checked += 1
            try:
                mask = font.getmask(char)
            except Exception:
                unsupported += 1
                continue
            bbox = mask.getbbox()
            if bbox is None:
                unsupported += 1
                continue
            if (
                (bbox, bytes(mask)) in missing_glyph_masks
                and ("\u3000" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff")
            ):
                unsupported += 1

        if checked == 0:
            return True
        return unsupported == 0

    @staticmethod
    def _font_file_exists(candidate: str) -> bool:
        path = Path(candidate)
        if path.is_file():
            return True
        try:
            ImageFont.truetype(candidate, size=12)
            return True
        except OSError:
            return False

    @staticmethod
    def _unique_candidates(candidates: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for candidate in candidates:
            key = safe_text(candidate).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @staticmethod
    def _latin_font_candidates() -> list[str]:
        return [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Regular.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]

    @classmethod
    def _japanese_font_candidates(cls) -> list[str]:
        return [
            r"C:\Windows\Fonts\YuGothR.ttc",
            r"C:\Windows\Fonts\YuGothM.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            r"C:\Windows\Fonts\meiryo.ttc",
            *cls._cjk_font_candidates(),
        ]

    @classmethod
    def _chinese_font_candidates(cls) -> list[str]:
        return [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            *cls._cjk_font_candidates(),
        ]

    @staticmethod
    def _cjk_font_candidates() -> list[str]:
        return [
            "NotoSansCJK-Regular.ttc",
            "NotoSansCJKjp-Regular.otf",
            "NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
        ]

    @staticmethod
    def _is_japanese_language(target_lang: str) -> bool:
        target = safe_text(target_lang).lower()
        return target in {"ja", "jp", "jpn", "japan", "japanese", "jpn_jpan"} or target.startswith("ja")

    @staticmethod
    def _is_chinese_language(target_lang: str) -> bool:
        target = safe_text(target_lang).lower()
        return target in {"zh", "zh-cn", "zh_cn", "ch", "china", "chinese", "zho_hans"} or target.startswith("zh")

    @classmethod
    def _is_cjk_language(cls, target_lang: str) -> bool:
        return cls._is_japanese_language(target_lang) or cls._is_chinese_language(target_lang)

    @staticmethod
    def _contains_japanese(text: str) -> bool:
        return any("\u3040" <= char <= "\u30ff" or "\u31f0" <= char <= "\u31ff" for char in safe_text(text))

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf" for char in safe_text(text))

    @classmethod
    def _contains_cjk(cls, text: str) -> bool:
        return cls._contains_japanese(text) or cls._contains_chinese(text)

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
            suffix = "" if self._contains_cjk(remaining) else "-"

            while split_at > 1 and self._text_width(draw, remaining[:split_at] + suffix, font) > max_width:
                split_at -= 1

            if split_at <= 1:
                return lines, remaining

            lines.append(remaining[:split_at] + suffix)
            remaining = remaining[split_at:]

        return lines, remaining

    def _balance_lines(
        self,
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        font,
        max_width: int,
    ) -> list[str]:
        """
        Tenta evitar primeira linha muito comprida e última linha muito curta.
        Isso melhora o visual centralizado dentro do balão.
        """

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
