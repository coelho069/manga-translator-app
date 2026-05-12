from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.cleaner import BubbleCleaner
from app.config import AppConfig
from app.detector import BubbleDetector
from app.ocr import OCRReader
from app.renderer import TextRenderer
from app.translator import BaseTranslator, GoogleTextTranslator
from app.types import AppResult, SpeechBubble
from app.utils import ensure_dir, read_image_cv2, safe_text, write_image_cv2


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
        output_dir = ensure_dir(output_dir or self.config.output_dir)
        self.debug_dir = ensure_dir(self.config.debug_dir or Path(output_dir) / "debug")

        self._progress(progress_callback, 0.05, "carregando imagem")
        original_image_path = Path(image_path)
        image = read_image_cv2(original_image_path)

        self._progress(progress_callback, 0.16, "detectando baloes")
        bubbles = self.detector.detect(image)

        total = max(1, len(bubbles))
        for index, bubble in enumerate(bubbles):
            bubble_progress = index / total
            self._progress(progress_callback, 0.25 + bubble_progress * 0.20, "lendo texto")
            crop = image[bubble.bbox.y1 : bubble.bbox.y2, bubble.bbox.x1 : bubble.bbox.x2]
            bubble.ocr_boxes = self.ocr.read(crop, offset_x=bubble.bbox.x1, offset_y=bubble.bbox.y1)
            bubble.source_text = safe_text(" ".join(safe_text(box.text) for box in bubble.ocr_boxes))

            if not bubble.source_text:
                bubble.processing_notes.append("sem texto OCR; balao mantido intacto")
                continue

            self._progress(progress_callback, 0.45 + bubble_progress * 0.15, "traduzindo")
            bubble.translated_text = safe_text(self.translator.translate(bubble.source_text))
            if not bubble.translated_text:
                bubble.processing_notes.append("sem traducao; balao mantido intacto")

        work_image = image.copy()
        cleaned_bubble_ids: set[int] = set()

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

        self._save_debug_image("debug_after_cleanup.png", work_image)
        self._save_debug_image("after_cleanup.png", work_image)

        for index, bubble in enumerate(bubbles):
            bubble_progress = index / total
            if bubble.source_text and bubble.translated_text and bubble.id in cleaned_bubble_ids:
                self._progress(progress_callback, 0.80 + bubble_progress * 0.14, "inserindo traducao")
                work_image = self.renderer.render(work_image, bubble)

        if not bubbles:
            self._progress(progress_callback, 0.80, "nenhum balao detectado")

        self._progress(progress_callback, 0.96, "finalizando")
        self._save_debug_image("debug_final_rendered.png", work_image)
        self._save_debug_image("final.png", work_image)
        self._save_debug_image("final_with_bbox.png", self._draw_debug_bboxes(work_image, bubbles))

        output_path = Path(output_dir) / f"translated_{original_image_path.stem}.png"
        write_image_cv2(output_path, work_image)
        self._progress(progress_callback, 1.0, "finalizado")

        return AppResult(
            original_image_path=original_image_path,
            translated_image_path=output_path,
            bubbles=bubbles,
            output_dir=Path(output_dir),
            metadata={
                "bubble_count": len(bubbles),
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

    def _save_bubble_debug(self, bubble: SpeechBubble) -> None:
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
