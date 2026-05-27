from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.cleaner import BubbleCleaner
from app.config import AppConfig
from app.detector import BubbleDetector
from app.ocr import OCRReader
from app.renderer import TextRenderer
from app.translator import (
    BaseTranslator,
    build_translator,
    is_translation_valid,
    normalize_lang_code,
    normalize_punctuation,
    preserve_terminal_punctuation,
)
from app.types import AppResult, BoundingBox, SpeechBubble
from app.utils import ensure_dir, read_image_cv2, safe_text, write_image_cv2


class MangaTranslatorPipeline:
    PROGRESS_STAGE_MAP = {
        "carregando imagem": ("load_image", "Carregando imagem"),
        "detectando baloes": ("detect", "Detectando baloes"),
        "lendo texto": ("ocr", "Lendo textos"),
        "traduzindo": ("translate", "Traduzindo falas"),
        "apagando texto original": ("clean", "Limpando baloes"),
        "inserindo traducao": ("render", "Inserindo traducao"),
        "nenhum balao detectado": ("finish", "Finalizando"),
        "finalizando": ("finish", "Finalizando"),
        "finalizado": ("finish", "Resultado pronto"),
    }

    def __init__(self, config: AppConfig | None = None, translator: BaseTranslator | None = None):
        self.config = config or AppConfig()
        ensure_dir(self.config.output_dir)
        self.detector = BubbleDetector(self.config)
        self.ocr = OCRReader(self.config)
        self.translator = translator or build_translator(self.config)
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
        if self.config.debug_enabled:
            self.debug_dir = ensure_dir(self.config.debug_dir or Path(output_dir) / "debug")
        self.renderer.debug_dir = self.debug_dir

        flow_report_path: Path | None = None
        if self.debug_dir is not None:
            flow_report_path = Path(self.debug_dir) / "bubble_flow_report.json"

        print(f"[MODO] {safe_text(self.config.translation_mode) or 'en_to_pt'}")
        print(f"[OCR] {safe_text(self.config.ocr_lang) or 'en'}")
        print(f"[TRADUCAO] {safe_text(self.config.source_lang) or 'en'} -> {safe_text(self.config.target_lang) or 'pt'}")

        self._progress(progress_callback, 0.05, "carregando imagem")
        step_start = perf_counter()
        original_image_path = Path(image_path)
        image = read_image_cv2(original_image_path)
        timings["load_image"] = self._elapsed(step_start)

        self._progress(progress_callback, 0.16, "detectando baloes")
        step_start = perf_counter()
        bubbles = self.detector.detect(image, page_label=original_image_path.name)
        timings["detect"] = self._elapsed(step_start)

        total = max(1, len(bubbles))
        flow_report = {bubble.id: self._new_flow_record(bubble) for bubble in bubbles}

        step_start = perf_counter()
        for index, bubble in enumerate(bubbles):
            bubble_progress = index / total
            self._progress(progress_callback, 0.25 + bubble_progress * 0.20, "lendo texto")
            record = flow_report[bubble.id]
            record["ocr_ran"] = True

            try:
                ocr_bbox = self._ocr_bbox_for_bubble(bubble, image.shape[:2])
                bubble.ocr_bbox = ocr_bbox
                print(
                    f"[OCR_BBOX] balloon={bubble.id} "
                    f"bbox=({ocr_bbox.x1},{ocr_bbox.y1},{ocr_bbox.x2},{ocr_bbox.y2})"
                )
                raw_crop = self._masked_crop_for_ocr(image, bubble, ocr_bbox)
                if raw_crop is None or raw_crop.size == 0:
                    record["skipped_reason"] = "crop vazio para OCR"
                    bubble.processing_notes.append("crop vazio para OCR")
                    continue

                crop, scale = self._resize_crop_for_ocr(raw_crop)
                bubble.ocr_boxes = self.ocr.read(crop, offset_x=ocr_bbox.x1, offset_y=ocr_bbox.y1)
                if scale != 1.0:
                    self._rescale_ocr_boxes(bubble, scale, ocr_bbox)

                ocr_texts = [safe_text(box.text) for box in bubble.ocr_boxes if safe_text(box.text)]
                if safe_text(self.config.source_lang).lower() == "ja":
                    bubble.source_text = safe_text("".join(ocr_texts))
                else:
                    bubble.source_text = safe_text(" ".join(ocr_texts))
            except Exception as exc:
                record["skipped_reason"] = f"erro no OCR: {exc}"
                bubble.processing_notes.append(f"erro no OCR: {exc}")
                continue

            record["source_text"] = bubble.source_text
            if not bubble.source_text:
                record["skipped_reason"] = "OCR vazio"
                bubble.processing_notes.append("sem texto OCR; balao mantido intacto")
                continue
        timings["ocr"] = self._elapsed(step_start)

        translatable_bubbles = [bubble for bubble in bubbles if bubble.source_text]
        page_texts = [bubble.source_text for bubble in translatable_bubbles]

        step_start = perf_counter()
        total_translatable = len(translatable_bubbles)
        translated_ok = 0
        translation_failed = 0
        src_for_log = normalize_lang_code(getattr(self.config, "source_lang", "en"), "auto") or "auto"
        tgt_for_log = normalize_lang_code(getattr(self.config, "target_lang", "pt"), "pt") or "pt"

        if page_texts:
            self._progress(progress_callback, 0.48, "traduzindo")
            for bubble in translatable_bubbles:
                record = flow_report[bubble.id]
                record["translation_ran"] = True
                source_text = bubble.source_text
                in_snippet = source_text[:120].replace('"', "'")
                print(
                    f'[TRANSLATION_INPUT] balloon={bubble.id} '
                    f'source={src_for_log} target={tgt_for_log} text="{in_snippet}"'
                )

                try:
                    translated_raw = self.translator.translate_text(
                        source_text,
                        source_lang=src_for_log,
                        target_lang=tgt_for_log,
                    )
                except Exception as exc:
                    translation_failed += 1
                    bubble.translated_text = ""
                    record["skipped_reason"] = f"erro na traducao: {exc}"
                    bubble.processing_notes.append(f"erro na traducao: {exc}")
                    print(f"[TRANSLATION_ERROR] balloon={bubble.id} error={exc}")
                    continue

                translated = normalize_punctuation(translated_raw, tgt_for_log)
                translated = preserve_terminal_punctuation(source_text, translated, tgt_for_log)
                out_snippet = translated[:120].replace('"', "'")
                print(f'[TRANSLATION_OUTPUT] balloon={bubble.id} text="{out_snippet}"')

                valid, reason = is_translation_valid(source_text, translated, tgt_for_log)
                print(f"[TRANSLATION_VALIDATE] balloon={bubble.id} valid={valid} reason={reason}")
                if not valid:
                    translation_failed += 1
                    bubble.translated_text = ""
                    record["skipped_reason"] = f"traducao invalida: {reason}"
                    bubble.processing_notes.append(f"traducao invalida: {reason}")
                    continue

                bubble.translated_text = translated
                record["translated_text"] = translated
                translated_ok += 1

        print(
            f"[TRANSLATION_SUMMARY] total={total_translatable} "
            f"translated={translated_ok} failed={translation_failed}"
        )
        timings["translate"] = self._elapsed(step_start)

        work_image = image.copy()
        cleaned_bubble_ids: set[int] = set()
        clean_elapsed = 0.0
        render_elapsed = 0.0

        if self.debug_dir is not None:
            self._save_debug_image("debug_before_cleanup.png", work_image)

        bubble_masks_norm = self._build_normalized_masks(bubbles, image.shape[:2])

        for index, bubble in enumerate(bubbles):
            if not (bubble.source_text and bubble.translated_text):
                continue

            bubble_progress = index / total
            record = flow_report[bubble.id]
            record["cleanup_ran"] = True
            bubble.cleanup_success = False

            own_mask = bubble_masks_norm[index] if index < len(bubble_masks_norm) else None
            others_union = np.zeros(image.shape[:2], dtype=np.uint8)
            for j, bm in enumerate(bubble_masks_norm):
                if j == index or bm is None:
                    continue
                if own_mask is not None and cv2.countNonZero(cv2.bitwise_and(own_mask, bm)) > 0:
                    print(
                        f"[CLEANUP_COLLISION] balloon={bubble.id} "
                        f"other={bubbles[j].id} action=reduce_margin"
                    )
                others_union = cv2.bitwise_or(others_union, bm)
            exclusion_mask = others_union

            self._progress(progress_callback, 0.62 + bubble_progress * 0.16, "apagando texto original")
            clean_start = perf_counter()
            try:
                work_image = self.cleaner.clean(work_image, bubble, exclusion_mask=exclusion_mask)
            except Exception as exc:
                bubble.processing_notes.append(f"erro na limpeza: {exc}")
                record["skipped_reason"] = f"erro na limpeza: {exc}"
                self.last_block_reason = str(exc)
                print(f"[CLEANUP_ERROR] balloon={bubble.id} error={exc}")
            clean_elapsed += self._elapsed(clean_start)

            if not self.cleaner.last_cleaned and bool(getattr(self.config, "force_clean_on_failed_mask", True)):
                print(
                    f"[CLEANUP_FALLBACK] balloon={bubble.id} method=fill_inside_mask "
                    f"reason={self.cleaner.last_block_reason or 'cleanup_failed'}"
                )
                fallback_start = perf_counter()
                try:
                    work_image = self.cleaner.force_clean_bubble_inner_area(
                        work_image, bubble, exclusion_mask=exclusion_mask
                    )
                except Exception as exc:
                    bubble.processing_notes.append(f"erro no fallback de limpeza: {exc}")
                    if not record["skipped_reason"]:
                        record["skipped_reason"] = f"erro no fallback de limpeza: {exc}"
                clean_elapsed += self._elapsed(fallback_start)

            if self.debug_dir is not None:
                self._save_bubble_debug(bubble)

            if self.cleaner.last_cleaned:
                bubble.cleanup_success = True
                cleaned_bubble_ids.add(bubble.id)
                print(f"[PIPELINE] Bubble {bubble.id}: limpeza OK - renderizacao PERMITIDA")
                if self.debug_dir is not None:
                    print(f"[PIPELINE] Bubble {bubble.id}: residual dark pixels = {self.cleaner.last_residual_dark_pixels}")
                    print(f"[PIPELINE] Bubble {bubble.id}: residual ratio = {self.cleaner.last_residual_ratio:.1%}")
                    self._save_debug_image(
                        f"debug_bubble_{bubble.id}_render_allowed.png", work_image
                    )
            else:
                block_reason = self.cleaner.last_block_reason or "limpeza falhou"
                bubble.processing_notes.append(f"limpeza falhou: {block_reason}; traducao nao renderizada")
                if not record["skipped_reason"]:
                    record["skipped_reason"] = f"limpeza falhou: {block_reason}"
                print(f"[PIPELINE] Bubble {bubble.id}: BLOQUEADO - {block_reason}")
                print(f"[PIPELINE] Bubble {bubble.id}: traducao NAO renderizada")
                if self.debug_dir is not None:
                    self._save_debug_image(
                        f"debug_bubble_{bubble.id}_render_blocked.png", work_image
                    )
                continue

            self._progress(progress_callback, 0.80 + bubble_progress * 0.14, "inserindo traducao")
            render_start = perf_counter()

            if bubble.id not in cleaned_bubble_ids:
                print(f"[PIPELINE] Bubble {bubble.id}: BLOQUEADO - id nao esta em cleaned_bubble_ids")
                bubble.processing_notes.append("renderizacao ignorada: id nao esta em cleaned_bubble_ids")
                if not record["skipped_reason"]:
                    record["skipped_reason"] = "id nao esta em cleaned_bubble_ids"
                continue

            if not bubble.cleanup_success:
                print(f"[PIPELINE] Bubble {bubble.id}: BLOQUEADO - cleanup_success=False")
                bubble.processing_notes.append("renderizacao ignorada: cleanup_success=False")
                if not record["skipped_reason"]:
                    record["skipped_reason"] = "cleanup_success=False"
                continue

            render_rect = self.renderer.get_safe_text_rect(bubble, work_image.shape)
            if render_rect is not None:
                rx, ry, rw, rh = render_rect
                bubble.render_bbox = BoundingBox(rx, ry, rx + rw, ry + rh)
                print(f"[RENDER_BBOX] balloon={bubble.id} bbox=({rx},{ry},{rx + rw},{ry + rh})")
            print(f"[RENDER] draw_after_cleanup balloon={bubble.id}")
            try:
                work_image = self.renderer.render(work_image, bubble)
                if bool(getattr(bubble, "render_success", False)):
                    record["render_ran"] = True
                    print(f"[PIPELINE] Bubble {bubble.id}: traducao renderizada com sucesso")
                    print(f"[RENDER_SUCCESS] balloon={bubble.id}")
                else:
                    last_note = bubble.processing_notes[-1] if bubble.processing_notes else "render_returned_no_draw"
                    if not record["skipped_reason"]:
                        record["skipped_reason"] = f"render falhou: {last_note}"
            except Exception as exc:
                if not record["skipped_reason"]:
                    record["skipped_reason"] = f"erro na renderizacao: {exc}"
                bubble.processing_notes.append(f"erro na renderizacao: {exc}")
            render_elapsed += self._elapsed(render_start)

        timings["clean"] = round(clean_elapsed, 4)
        if self.debug_dir is not None:
            self._save_debug_image("debug_after_cleanup.png", work_image)
            self._save_debug_image("after_cleanup.png", work_image)
        timings["render"] = round(render_elapsed, 4)

        if not bubbles:
            self._progress(progress_callback, 0.80, "nenhum balao detectado")

        self._progress(progress_callback, 0.96, "finalizando")
        if self.debug_dir is not None:
            self._save_debug_image("debug_final_rendered.png", work_image)
            self._save_debug_image("final.png", work_image)
            self._save_debug_image("final_with_bbox.png", self._draw_debug_bboxes(work_image, bubbles))

        flow_report_list = self._finalize_flow_report(bubbles, flow_report)
        if flow_report_path is not None and self.debug_dir is not None:
            self._save_flow_report(flow_report_path, flow_report_list)

        detected_count = len(bubbles)
        ocr_count = sum(1 for bubble in bubbles if bubble.source_text)
        translated_count = sum(1 for bubble in bubbles if bubble.translated_text)
        rendered_count = sum(1 for row in flow_report_list if row.get("render_ran"))
        cleanup_success_count = len(cleaned_bubble_ids)

        output_path = Path(output_dir) / f"translated_{original_image_path.stem}.png"
        step_start = perf_counter()
        write_image_cv2(output_path, work_image)
        timings["save"] = self._elapsed(step_start)
        timings["total"] = self._elapsed(total_start)
        self._progress(progress_callback, 1.0, "finalizado")

        self._print_summary(bubbles, cleaned_bubble_ids, flow_report_list)

        return AppResult(
            original_image_path=original_image_path,
            translated_image_path=output_path,
            bubbles=bubbles,
            output_dir=Path(output_dir),
            metadata={
                "bubble_count": len(bubbles),
                "translation_mode": safe_text(self.config.translation_mode) or "en_to_pt",
                "source_lang": safe_text(self.config.source_lang) or "en",
                "ocr_lang": safe_text(self.config.ocr_lang) or "en",
                "target_lang": safe_text(self.config.target_lang) or "pt",
                "translation_style": safe_text(self.config.translation_style) or "natural",
                "performance_mode": safe_text(self.config.performance_mode) or "balanced",
                "timings": timings,
                "detected_count": detected_count,
                "ocr_count": ocr_count,
                "translated_count": translated_count,
                "rendered_count": rendered_count,
                "cleanup_success_count": cleanup_success_count,
                "total_time": timings.get("total", 0.0),
                "bubble_flow_report_path": str(flow_report_path) if flow_report_path is not None else "",
                "cleaned_bubble_ids": sorted(cleaned_bubble_ids),
                "skipped_bubbles": [
                    {"id": bubble.id, "notes": list(bubble.processing_notes)}
                    for bubble in bubbles
                    if bubble.processing_notes
                ],
                "debug_dir": str(self.debug_dir) if self.debug_dir is not None else "",
            },
        )

    def _print_summary(
        self,
        bubbles: list[SpeechBubble],
        cleaned_bubble_ids: set[int],
        flow_report_list: list[dict],
    ) -> None:
        print("\n" + "=" * 60)
        print("[RESUMO] Resultado da traducao")
        print("=" * 60)
        print(f"  Baloes detectados: {len(bubbles)}")
        print(f"  Limpezas bem-sucedidas: {len(cleaned_bubble_ids)}")
        print(f"  Renderizacoes: {sum(1 for r in flow_report_list if r.get('render_ran'))}")

        blocked = [r for r in flow_report_list if r.get("skipped_reason")]
        if blocked:
            print(f"  Baloes bloqueados: {len(blocked)}")
            for r in blocked:
                print(f"    - Bubble {r['id']}: {r['skipped_reason']}")

        print("=" * 60 + "\n")

    @classmethod
    def _progress(cls, callback, value: float, message: str) -> None:
        if callback is None:
            return
        clamped = float(max(0.0, min(1.0, value)))
        safe_message = safe_text(message)
        stage_id, stage_label = cls.PROGRESS_STAGE_MAP.get(
            safe_message.lower(),
            ("processing", safe_message.capitalize() or "Processando"),
        )
        meta = {
            "stage_id": stage_id,
            "stage_label": stage_label,
            "stage_status": "done" if clamped >= 1.0 else "running",
        }
        try:
            callback(clamped, safe_message, meta)
        except TypeError:
            callback(clamped, safe_message)

    @staticmethod
    def _elapsed(start: float) -> float:
        return round(perf_counter() - start, 4)

    @staticmethod
    def _ocr_bbox_for_bubble(bubble: SpeechBubble, shape: tuple[int, int]) -> BoundingBox:
        image_h, image_w = shape[:2]
        margin = max(2, min(12, int(max(bubble.bbox.width, bubble.bbox.height) * 0.04)))
        return BoundingBox(
            x1=max(0, bubble.bbox.x1 - margin),
            y1=max(0, bubble.bbox.y1 - margin),
            x2=min(image_w, bubble.bbox.x2 + margin),
            y2=min(image_h, bubble.bbox.y2 + margin),
        )

    @staticmethod
    def _masked_crop_for_ocr(image: np.ndarray, bubble: SpeechBubble, ocr_bbox: BoundingBox) -> np.ndarray:
        crop = image[ocr_bbox.y1 : ocr_bbox.y2, ocr_bbox.x1 : ocr_bbox.x2]
        if crop is None or crop.size == 0:
            return crop

        if getattr(bubble, "mask", None) is None or bubble.mask.size == 0:
            return crop

        mask_crop = bubble.mask[ocr_bbox.y1 : ocr_bbox.y2, ocr_bbox.x1 : ocr_bbox.x2]
        if mask_crop is None or mask_crop.size == 0:
            return crop
        if mask_crop.shape[:2] != crop.shape[:2]:
            return crop

        masked = np.full_like(crop, 255)
        masked[mask_crop > 0] = crop[mask_crop > 0]
        return masked

    def _resize_crop_for_ocr(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = crop.shape[:2]
        min_side = min(height, width)
        scale = 1.0
        if min_side < 96:
            scale = min(4.0, 96 / max(1, min_side))
        elif min_side < 140:
            scale = min(2.0, 140 / max(1, min_side))

        if scale <= 1.0:
            return crop, 1.0

        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized, scale

    @staticmethod
    def _rescale_ocr_boxes(bubble: SpeechBubble, scale: float, ocr_bbox: BoundingBox) -> None:
        if scale <= 0 or scale == 1.0:
            return
        ox = ocr_bbox.x1
        oy = ocr_bbox.y1
        for box in bubble.ocr_boxes:
            scaled_polygon = []
            for x, y in box.polygon:
                local_x = (x - ox) / scale
                local_y = (y - oy) / scale
                scaled_polygon.append((int(round(ox + local_x)), int(round(oy + local_y))))
            box.polygon = scaled_polygon

    @staticmethod
    def _new_flow_record(bubble: SpeechBubble) -> dict:
        return {
            "id": bubble.id,
            "bbox": {
                "x1": bubble.bbox.x1,
                "y1": bubble.bbox.y1,
                "x2": bubble.bbox.x2,
                "y2": bubble.bbox.y2,
            },
            "detected": True,
            "ocr_ran": False,
            "source_text": "",
            "translation_ran": False,
            "translated_text": "",
            "cleanup_ran": False,
            "render_ran": False,
            "skipped_reason": "",
        }

    @staticmethod
    def _finalize_flow_report(bubbles: list[SpeechBubble], flow_report: dict[int, dict]) -> list[dict]:
        rows = []
        for bubble in bubbles:
            row = flow_report.get(bubble.id) or MangaTranslatorPipeline._new_flow_record(bubble)
            row["source_text"] = bubble.source_text
            row["translated_text"] = bubble.translated_text
            if not row["skipped_reason"] and bubble.processing_notes:
                row["skipped_reason"] = "; ".join(bubble.processing_notes)
            rows.append(row)
        return rows

    @staticmethod
    def _save_flow_report(path: Path, rows: list[dict]) -> None:
        ensure_dir(path.parent)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

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

        self._save_debug_image(
            f"bubble_{bubble.id}_debug_before_cleanup.png",
            self.cleaner.last_before_image,
        )
        self._save_debug_image(
            f"bubble_{bubble.id}_debug_after_cleanup.png",
            self.cleaner.last_after_image,
        )

    def _save_debug_image(self, filename: str, image: np.ndarray | None) -> None:
        if self.debug_dir is None or image is None or image.size == 0:
            return
        write_image_cv2(Path(self.debug_dir) / filename, image)

    @staticmethod
    def _build_normalized_masks(
        bubbles: list[SpeechBubble], shape: tuple[int, int]
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for b in bubbles:
            bm = getattr(b, "mask", None)
            normalized = np.zeros(shape, dtype=np.uint8)
            if bm is not None and bm.size > 0:
                h = min(shape[0], bm.shape[0])
                w = min(shape[1], bm.shape[1])
                normalized[:h, :w] = (bm[:h, :w] > 0).astype(np.uint8) * 255
            else:
                x1 = max(0, min(shape[1], b.bbox.x1))
                y1 = max(0, min(shape[0], b.bbox.y1))
                x2 = max(0, min(shape[1], b.bbox.x2))
                y2 = max(0, min(shape[0], b.bbox.y2))
                if x2 > x1 and y2 > y1:
                    normalized[y1:y2, x1:x2] = 255
            masks.append(normalized)
        return masks

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
