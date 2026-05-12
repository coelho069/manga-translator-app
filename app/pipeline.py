from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.cleaner import BubbleCleaner
from app.config import AppConfig
from app.detector import BubbleDetector
from app.ocr import OCRReader
from app.renderer import TextRenderer
from app.translator import BaseTranslator, GoogleTextTranslator
from app.types import AppResult, SpeechBubble
from app.utils import ensure_dir, read_image_cv2, resolve_ocr_lang, resolve_translation_lang, safe_text, write_image_cv2


class MangaTranslatorPipeline:
    def __init__(self, config: AppConfig | None = None, translator: BaseTranslator | None = None):
        self.config = config or AppConfig()
        ensure_dir(self.config.output_dir)
        self.detector = BubbleDetector(self.config)
        self.ocr = OCRReader(self.config)
        self.translator = translator or GoogleTextTranslator(self.config)
        self.cleaner = BubbleCleaner(self.config)
        self.renderer = TextRenderer(self.config)
        self.debug_dir: Path | None = None

    def run(
        self,
        image_path: Path | str,
        output_dir: Path | str | None = None,
        progress_callback=None,
    ) -> AppResult:
        total_start = perf_counter()
        timings: dict[str, float] = {}
        output_dir = ensure_dir(output_dir or self.config.output_dir)
        self.debug_dir = None
        if self.config.debug_enabled or self.config.debug_dir is not None:
            self.debug_dir = ensure_dir(self.config.debug_dir or Path(output_dir) / "debug")

        self._progress(progress_callback, 0.05, "carregando imagem")
        step_start = perf_counter()
        original_image_path = Path(image_path)
        image = read_image_cv2(original_image_path)
        timings["load_image"] = self._elapsed(step_start)

        self._progress(progress_callback, 0.16, "detectando baloes")
        step_start = perf_counter()
        bubbles = self.detector.detect(image)
        timings["detect"] = self._elapsed(step_start)

        total = max(1, len(bubbles))
        step_start = perf_counter()
        for index, bubble in enumerate(bubbles):
            bubble_progress = index / total
            self._progress(progress_callback, 0.25 + bubble_progress * 0.20, "lendo texto")
            crop = image[bubble.bbox.y1 : bubble.bbox.y2, bubble.bbox.x1 : bubble.bbox.x2]
            if not self._should_run_ocr(bubble, crop):
                bubble.processing_notes.append("OCR pulado por area pequena ou sem texto provavel")
                continue
            bubble.ocr_boxes = self.ocr.read(crop, offset_x=bubble.bbox.x1, offset_y=bubble.bbox.y1)
            bubble.source_text = safe_text(" ".join(safe_text(box.text) for box in bubble.ocr_boxes))

            if not bubble.source_text:
                bubble.processing_notes.append("sem texto OCR; balao mantido intacto")
                continue
        timings["ocr"] = self._elapsed(step_start)

        translatable_bubbles = [bubble for bubble in bubbles if bubble.source_text]
        page_texts = [bubble.source_text for bubble in translatable_bubbles]
        step_start = perf_counter()
        if page_texts:
            self._progress(progress_callback, 0.48, "traduzindo")
            translations = self.translator.translate_batch(page_texts)
            for bubble, translation in zip(translatable_bubbles, translations):
                bubble.translated_text = safe_text(translation)
                if not bubble.translated_text:
                    bubble.processing_notes.append("sem traducao; balao mantido intacto")
        timings["translate"] = self._elapsed(step_start)

        work_image = image.copy()
        cleaned_bubble_ids: set[int] = set()

        step_start = perf_counter()
        for index, bubble in enumerate(bubbles):
            bubble_progress = index / total
            if bubble.source_text and bubble.translated_text:
                self._progress(progress_callback, 0.62 + bubble_progress * 0.16, "apagando texto original")
                bubble.cleanup_success = False
                work_image = self.cleaner.clean(work_image, bubble)
                self._save_bubble_debug(bubble)

                if self.cleaner.last_cleaned:
                    bubble.cleanup_success = True
                    cleaned_bubble_ids.add(bubble.id)
                else:
                    bubble.processing_notes.append("limpeza falhou; traducao nao renderizada")
        timings["clean"] = self._elapsed(step_start)

        self._save_debug_image("debug_after_cleanup.png", work_image)
        self._save_debug_image("after_cleanup.png", work_image)

        step_start = perf_counter()
        for index, bubble in enumerate(bubbles):
            bubble_progress = index / total
            if bubble.source_text and bubble.translated_text and bubble.id in cleaned_bubble_ids:
                self._progress(progress_callback, 0.80 + bubble_progress * 0.14, "inserindo traducao")
                work_image = self.renderer.render(work_image, bubble)
        timings["render"] = self._elapsed(step_start)

        if not bubbles:
            self._progress(progress_callback, 0.80, "nenhum balao detectado")

        self._progress(progress_callback, 0.96, "finalizando")
        self._save_debug_image("debug_final_rendered.png", work_image)
        self._save_debug_image("final.png", work_image)
        self._save_debug_image("final_with_bbox.png", self._draw_debug_bboxes(work_image, bubbles))

        output_path = Path(output_dir) / f"translated_{original_image_path.stem}.png"
        step_start = perf_counter()
        write_image_cv2(output_path, work_image)
        timings["save"] = self._elapsed(step_start)
        timings["total"] = self._elapsed(total_start)
        self._progress(progress_callback, 1.0, "finalizado")

        return AppResult(
            original_image_path=original_image_path,
            translated_image_path=output_path,
            bubbles=bubbles,
            output_dir=Path(output_dir),
            metadata={
                "bubble_count": len(bubbles),
                "source_lang": resolve_translation_lang(self.config.source_lang),
                "ocr_lang": resolve_ocr_lang(self.config.ocr_lang or self.config.source_lang),
                "target_lang": safe_text(self.config.target_lang) or "pt",
                "translation_style": safe_text(self.config.translation_style) or "natural",
                "performance_mode": safe_text(self.config.performance_mode) or "balanced",
                "timings": timings,
                "cleaned_bubble_ids": sorted(cleaned_bubble_ids),
                "skipped_bubbles": [
                    {"id": bubble.id, "notes": list(bubble.processing_notes)}
                    for bubble in bubbles
                    if bubble.processing_notes
                ],
                "debug_dir": str(self.debug_dir) if self.debug_dir is not None else "",
            },
        )

    @staticmethod
    def _progress(callback, value: float, message: str) -> None:
        if callback is not None:
            callback(float(max(0.0, min(1.0, value))), message)

    @staticmethod
    def _elapsed(start: float) -> float:
        return round(perf_counter() - start, 4)

    def _should_run_ocr(self, bubble: SpeechBubble, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        if bubble.bbox.width < self.config.min_ocr_width or bubble.bbox.height < self.config.min_ocr_height:
            return False
        if bubble.bbox.width * bubble.bbox.height < self.config.min_ocr_area:
            return False
        if self.config.performance_mode == "quality":
            return True

        mask_crop = bubble.mask[bubble.bbox.y1 : bubble.bbox.y2, bubble.bbox.x1 : bubble.bbox.x2]
        if mask_crop.size == 0:
            return True
        inner = mask_crop.astype(np.uint8)
        erode_px = max(1, self.config.bubble_erode_px // 2)
        kernel_size = erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inner = cv2.erode(inner, kernel, iterations=1)
        if cv2.countNonZero(inner) == 0:
            inner = mask_crop.astype(np.uint8)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark = np.zeros_like(gray, dtype=np.uint8)
        dark[gray < self.config.dark_text_threshold] = 255
        dark = cv2.bitwise_and(dark, inner)
        min_dark_pixels = 8 if self.config.performance_mode == "fast" else 4
        return cv2.countNonZero(dark) >= min_dark_pixels

    def _save_bubble_debug(self, bubble: SpeechBubble) -> None:
        if self.debug_dir is None:
            return
        self._save_debug_image("debug_inner_mask.png", self.cleaner.last_inner_mask)
        self._save_debug_image("debug_dark_text_mask.png", self.cleaner.last_dark_mask)
        self._save_debug_image("debug_ocr_mask.png", self.cleaner.last_ocr_mask)
        self._save_debug_image("debug_final_cleanup_mask.png", self.cleaner.last_cleanup_mask)

        self._save_debug_image(f"bubble_{bubble.id}_inner_mask.png", self.cleaner.last_inner_mask)
        self._save_debug_image(f"bubble_{bubble.id}_dark_text_mask.png", self.cleaner.last_dark_mask)
        self._save_debug_image(f"bubble_{bubble.id}_ocr_mask.png", self.cleaner.last_ocr_mask)
        self._save_debug_image(f"bubble_{bubble.id}_final_cleanup_mask.png", self.cleaner.last_cleanup_mask)

    def _save_debug_image(self, filename: str, image: np.ndarray | None) -> None:
        if self.debug_dir is None or image is None or image.size == 0:
            return
        write_image_cv2(Path(self.debug_dir) / filename, image)

    @staticmethod
    def _draw_debug_bboxes(image: np.ndarray, bubbles: list[SpeechBubble]) -> np.ndarray:
        debug = image.copy()
        for bubble in bubbles:
            color = (0, 180, 0) if not bubble.processing_notes else (0, 165, 255)
            cv2.rectangle(debug, (bubble.bbox.x1, bubble.bbox.y1), (bubble.bbox.x2, bubble.bbox.y2), color, 2)
            cv2.putText(
                debug,
                str(bubble.id),
                (bubble.bbox.x1, max(0, bubble.bbox.y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return debug
