from __future__ import annotations

import json
import os
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
    ConfigurableApiTranslator,
    get_last_translation_debug,
    get_translation_settings,
    validate_translation_for_target,
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
        self.translator = translator or ConfigurableApiTranslator(self.config)
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
        page_timeout_seconds = max(60, int(os.getenv("PIPELINE_PAGE_TIMEOUT_SECONDS", "900") or "900"))
        deadline = total_start + page_timeout_seconds
        timings: dict[str, float] = {}
        output_dir = ensure_dir(output_dir or self.config.output_dir)

        self.debug_dir = None
        if self.config.debug_enabled or self.config.debug_dir is not None:
            self.debug_dir = ensure_dir(self.config.debug_dir or Path(output_dir) / "debug")

        flow_report_path: Path | None = None
        if self.debug_dir is not None:
            flow_report_path = Path(self.debug_dir) / "bubble_flow_report.json"

        translation_settings = get_translation_settings(self.config)
        print(
            f"[TRANSLATOR] provider={translation_settings.provider} model={translation_settings.model} "
            f"source={translation_settings.source_lang} target={translation_settings.target_lang}"
        )
        flow_label = safe_text(self.config.translation_flow) or f"{translation_settings.source_lang}_to_{translation_settings.target_lang}"
        print(f"[MODO] {flow_label}")
        print(f"[OCR] {safe_text(self.config.ocr_lang) or 'japan'}")
        print(f"[TRADUCAO] {translation_settings.source_lang} -> {translation_settings.target_lang}")

        self._check_timeout(deadline, "inicio")
        self._progress(progress_callback, 0.05, "carregando imagem")
        step_start = perf_counter()
        original_image_path = Path(image_path)
        page_label = original_image_path.stem
        image = read_image_cv2(original_image_path)
        self._record_timing(timings, "load_image", step_start)
        print("[FLOW] imagem carregada")

        self._check_timeout(deadline, "load_image")
        self._progress(progress_callback, 0.16, "detectando baloes")
        step_start = perf_counter()
        try:
            bubbles = self.detector.detect(image, page_label=page_label)
        except Exception as exc:
            print(f"[FLOW_ERROR] stage=detect page={page_label} balloon=0 error={exc}")
            bubbles = []
        if not bubbles:
            print(f"[FLOW_ERROR] stage=detect page={page_label} balloon=0 error=nenhum balao detectado; usando pagina inteira")
            bubbles = [self._full_page_bubble(image)]
        self._record_timing(timings, "detect", step_start)
        print(f"[FLOW] baloes detectados: {len(bubbles)}")

        self._check_timeout(deadline, "detect")
        total = max(1, len(bubbles))
        flow_report = {bubble.id: self._new_flow_record(bubble) for bubble in bubbles}

        step_start = perf_counter()
        print("[FLOW] ocr_start")
        print(f"[OCR] pipeline_enter page={page_label}")
        for index, bubble in enumerate(bubbles):
            self._check_timeout(deadline, f"ocr balao {bubble.id}")
            bubble_progress = index / total
            self._progress(progress_callback, 0.25 + bubble_progress * 0.20, "lendo texto")
            record = flow_report[bubble.id]
            record["ocr_ran"] = True
            print(f"[BALLOON] index={index + 1} id={bubble.id} bbox={bubble.bbox.x1},{bubble.bbox.y1},{bubble.bbox.x2},{bubble.bbox.y2}")
            print(f"[OCR] start page={page_label} balloon={bubble.id}")
            print(
                f"[OCR] crop_bbox page={page_label} balloon={bubble.id} "
                f"bbox={bubble.bbox.x1},{bubble.bbox.y1},{bubble.bbox.x2},{bubble.bbox.y2}"
            )

            try:
                bubble.ocr_boxes, ocr_texts, ocr_reason = self._read_ocr_with_fallbacks(image, bubble, page_label)
                # CJK scripts don't use whitespace between tokens; everything
                # else gets a single space separator.
                cjk_langs = {"ja", "zh", "ko"}
                if safe_text(self.config.source_lang).lower() in cjk_langs:
                    bubble.source_text = safe_text("".join(ocr_texts))
                else:
                    bubble.source_text = safe_text(" ".join(ocr_texts))
            except Exception as exc:
                record["skipped_reason"] = f"erro no OCR: {exc}"
                bubble.processing_notes.append(f"erro no OCR: {exc}")
                print(f"[OCR] falhou no balao {bubble.id}: {exc}")
                print(f"[OCR_ERROR] page={page_label} balloon={bubble.id} error={exc}")
                print(f"[FLOW_ERROR] stage=ocr page={page_label} balloon={bubble.id} error={exc}")
                print(f"[OCR_EMPTY] page={page_label} balloon={bubble.id} reason=erro no OCR: {exc}")
                continue

            record["source_text"] = bubble.source_text
            print(f"[OCR] result page={page_label} balloon={bubble.id} text=\"{bubble.source_text}\"")
            if not bubble.source_text:
                record["skipped_reason"] = ocr_reason or "OCR vazio"
                bubble.processing_notes.append(record["skipped_reason"])
                print(f"[OCR] nenhum texto detectado no balao {bubble.id}")
                print(f"[OCR_EMPTY] page={page_label} balloon={bubble.id} reason={record['skipped_reason']}")
                print(f"[FLOW_ERROR] stage=ocr page={page_label} balloon={bubble.id} error={record['skipped_reason']}")
                print(f"[FLOW_ERROR] ocr OCR vazio no balao {bubble.id}")
                continue
            print(f"[OCR] Bubble {bubble.id}: texto={bubble.source_text}")
        self._record_timing(timings, "ocr", step_start)
        ocr_detected_count = sum(1 for bubble in bubbles if bubble.source_text)
        ocr_empty_count = max(0, len(bubbles) - ocr_detected_count)
        print(
            f"[OCR_SUMMARY] page={page_label} total_balloons={len(bubbles)} "
            f"detected_texts={ocr_detected_count} empty={ocr_empty_count}"
        )
        print("[FLOW] ocr_done")
        print("[FLOW] OCR concluido")
        self._save_debug_image("debug_after_ocr.png", image)

        translatable_bubbles = [bubble for bubble in bubbles if bubble.source_text]

        step_start = perf_counter()
        print("[FLOW] translation_start")
        self._check_timeout(deadline, "ocr")
        job_source_lang = safe_text(translation_settings.source_lang) or "auto"
        job_target_lang = safe_text(translation_settings.target_lang) or "en"
        improved_count = 0
        if translatable_bubbles:
            self._progress(progress_callback, 0.48, "traduzindo")
            # Build the page context once — list of OCR strings for every
            # balloon that has text. Each balloon will receive a short
            # neighbourhood window (previous + next bubbles) so the refiner
            # can keep tone/coherence WITHOUT mixing dialogue lines.
            page_texts = [safe_text(b.source_text) for b in translatable_bubbles]
            for idx, bubble in enumerate(translatable_bubbles):
                record = flow_report[bubble.id]
                record["translation_ran"] = True
                # Keep the closest neighbours: 2 before + 2 after.
                start = max(0, idx - 2)
                end = min(len(page_texts), idx + 3)
                ctx_window = [page_texts[i] for i in range(start, end) if i != idx and page_texts[i]]
                translated = ""
                try:
                    # Per-balloon try/except: a single failure must never abort
                    # the page. We still keep the page failure only for the
                    # case where every balloon failed (handled later).
                    translated = safe_text(
                        self.translator.translate(
                            bubble.source_text,
                            balloon_id=bubble.id,
                            source_lang=job_source_lang,
                            target_lang=job_target_lang,
                            context=ctx_window,
                        )
                    )
                    debug_info = get_last_translation_debug()
                    print(f"[TRANSLATION_RAW] balloon={bubble.id} raw=\"{safe_text(debug_info.get('raw'))}\"")
                    print(f"[TRANSLATION_DECODED] balloon={bubble.id} text=\"{safe_text(debug_info.get('decoded'))}\"")
                    if not translated:
                        raise RuntimeError("traducao vazia")
                except Exception as exc:
                    print(f"[TRANSLATION_QUALITY_ERROR] balloon={bubble.id} error={exc}")
                    print(f"[TRANSLATION_ERROR] balloon={bubble.id} error={exc}")
                    print(f"[TRANSLATION_RAW] balloon={bubble.id} raw=\"\"")
                    print(f"[TRANSLATION_DECODED] balloon={bubble.id} text=\"\"")
                    print(
                        f"[TRANSLATION_VALIDATE] balloon={bubble.id} valid=false reason=error:{exc}"
                    )
                    record["skipped_reason"] = f"erro na traducao: {exc}"
                    bubble.processing_notes.append("sem traducao valida; balao mantido intacto")
                    print(f"[FLOW_ERROR] translation {exc} no balao {bubble.id}")
                    continue
                # Defence-in-depth: never let an empty / same-as-original /
                # still-in-source candidate slip past translate_text into the
                # renderer. The renderer would either fail or burn bad text
                # onto the page.
                valid, reason = validate_translation_for_target(
                    bubble.source_text, translated, job_source_lang, job_target_lang
                )
                if not valid:
                    print(
                        f"[TRANSLATION_QUALITY_VALIDATE] balloon={bubble.id} valid=false reason=pipeline:{reason}"
                    )
                    record["skipped_reason"] = f"traducao invalida: {reason}"
                    bubble.processing_notes.append(f"traducao invalida ({reason}); balao mantido intacto")
                    print(f"[FLOW_ERROR] translation invalida ({reason}) no balao {bubble.id}")
                    continue
                bubble.translated_text = translated
                record["translated_text"] = bubble.translated_text
                if reason in {"ok"}:
                    improved_count += 1
                print(f"[TRADUCAO] Bubble {bubble.id}: original={bubble.source_text}")
                print(f"[TRADUCAO] Bubble {bubble.id}: traduzido={bubble.translated_text}")
                print("[TRADUCAO] sucesso")
        self._record_timing(timings, "translate", step_start)
        print("[FLOW] translation_done")
        translated_ok = sum(1 for b in bubbles if b.translated_text)
        translation_failed = max(0, len(translatable_bubbles) - translated_ok)
        print(f"[FLOW] traducao concluida: {translated_ok}/{len(translatable_bubbles)} baloes traduzidos")
        print(
            f"[TRANSLATION_SUMMARY] total={len(translatable_bubbles)} translated={translated_ok} "
            f"improved={improved_count} failed={translation_failed} "
            f"source={job_source_lang} target={job_target_lang}"
        )
        if translatable_bubbles and translated_ok == 0:
            print("[FLOW] AVISO: todos os baloes falharam na traducao - verifique o motor de traducao")

        work_image = image.copy()
        self.cleaner.set_bubble_context(bubbles)
        self.renderer.set_bubble_context(bubbles)
        self.renderer.reset_occupancy(work_image.shape)
        cleaned_bubble_ids: set[int] = set()
        clean_elapsed = 0.0
        render_elapsed = 0.0

        editable_bubbles = [bubble for bubble in bubbles if bubble.source_text and bubble.translated_text]
        total_editable = max(1, len(editable_bubbles))
        print(f"[FLOW] baloes editaveis (OCR+traducao ok): {len(editable_bubbles)}")
        if not editable_bubbles and bubbles:
            print("[FLOW] AVISO: nenhum balao editavel — limpeza e renderizacao serao ignoradas")

        self._save_debug_image("debug_before_cleanup.png", work_image)
        print("[FLOW] cleanup_start")
        for index, bubble in enumerate(editable_bubbles):
            self._check_timeout(deadline, f"cleanup balao {bubble.id}")
            bubble_progress = index / total_editable
            record = flow_report[bubble.id]
            record["cleanup_ran"] = True
            bubble.cleanup_success = False

            self._progress(progress_callback, 0.62 + bubble_progress * 0.16, "apagando texto original")
            clean_start = perf_counter()
            print(f"[CLEANUP] start page={page_label} balloon={bubble.id}")
            print(f"[CLEANUP] start balloon={bubble.id}")
            # Each balloon is wrapped in try/except so that a failure in one
            # never prevents the remaining balloons from being processed.
            try:
                work_image = self.cleaner.clean(work_image, bubble)
            except Exception as exc:
                bubble.processing_notes.append(f"erro na limpeza: {exc}")
                record["skipped_reason"] = f"erro na limpeza: {exc}"
                print(f"[CLEANER] falha no balao {bubble.id}: {exc}")
                print(f"[CLEANUP_ERROR] page={page_label} balloon={bubble.id} error={exc}")
                print(f"[CLEANUP_ERROR] balloon={bubble.id} error={exc}")
                print(f"[FLOW_ERROR] cleanup {exc}")
            clean_elapsed += self._elapsed(clean_start)

            if not self.cleaner.last_cleaned and bool(getattr(self.config, "force_clean_on_failed_mask", True)):
                print(f"[CLEANUP_FALLBACK] page={page_label} balloon={bubble.id} reason=advanced_cleanup_failed")
                fallback_start = perf_counter()
                try:
                    work_image = self.cleaner.force_clean_bubble_inner_area(work_image, bubble)
                except Exception as exc:
                    bubble.processing_notes.append(f"erro no fallback de limpeza: {exc}")
                    if not record["skipped_reason"]:
                        record["skipped_reason"] = f"erro no fallback de limpeza: {exc}"
                    print(f"[CLEANER] falha no balao {bubble.id}: {exc}")
                    print(f"[CLEANUP_ERROR] page={page_label} balloon={bubble.id} error={exc}")
                    print(f"[CLEANUP_ERROR] balloon={bubble.id} error={exc}")
                    print(f"[FLOW_ERROR] cleanup {exc}")
                clean_elapsed += self._elapsed(fallback_start)

            if not self.cleaner.last_cleaned and bool(getattr(self.config, "force_clean_on_failed_mask", True)):
                print(f"[CLEANUP_FALLBACK] page={page_label} balloon={bubble.id} reason=force_cleanup_failed")
                simple_start = perf_counter()
                try:
                    work_image = self.cleaner.simple_cover_text_area(work_image, bubble)
                except Exception as exc:
                    bubble.processing_notes.append(f"erro no fallback simples de limpeza: {exc}")
                    print(f"[CLEANUP_ERROR] page={page_label} balloon={bubble.id} error={exc}")
                    print(f"[CLEANUP_ERROR] balloon={bubble.id} error={exc}")
                    print(f"[FLOW_ERROR] cleanup {exc}")
                clean_elapsed += self._elapsed(simple_start)

            # Prevent rendering on top of visible old text.
            if self.cleaner.last_cleaned and self.cleaner.has_visible_dark_residue(work_image, bubble):
                residue_start = perf_counter()
                try:
                    work_image = self.cleaner.force_clean_bubble_inner_area(work_image, bubble)
                except Exception as exc:
                    bubble.processing_notes.append(f"erro no cleanup de residuo: {exc}")
                    print(f"[CLEANUP_ERROR] balloon={bubble.id} error={exc}")
                    print(f"[FLOW_ERROR] cleanup {exc}")
                clean_elapsed += self._elapsed(residue_start)
                if self.cleaner.has_visible_dark_residue(work_image, bubble):
                    simple_start = perf_counter()
                    try:
                        work_image = self.cleaner.simple_cover_text_area(work_image, bubble)
                    except Exception as exc:
                        bubble.processing_notes.append(f"erro no fallback simples de residuo: {exc}")
                        print(f"[CLEANUP_ERROR] balloon={bubble.id} error={exc}")
                        print(f"[FLOW_ERROR] cleanup {exc}")
                    clean_elapsed += self._elapsed(simple_start)
                    if not self.cleaner.last_cleaned:
                        bubble.processing_notes.append("texto original ainda visivel apos fallback")
                        print(f"[CLEANER] falha no balao {bubble.id}: texto original ainda visivel")

            self._save_bubble_debug(bubble)

            if self.cleaner.last_cleaned:
                bubble.cleanup_success = True
                cleaned_bubble_ids.add(bubble.id)
                print(f"[CLEANUP] success page={page_label} balloon={bubble.id}")
            else:
                bubble.processing_notes.append("limpeza falhou; renderizacao ignorada neste balao")
                if not record["skipped_reason"]:
                    record["skipped_reason"] = "limpeza falhou"
                print(f"[CLEANER] falha no balao {bubble.id}")
                print(f"[CLEANUP_ERROR] page={page_label} balloon={bubble.id} error=limpeza falhou")
                print(f"[CLEANUP_ERROR] balloon={bubble.id} error=limpeza falhou")
                print(f"[FLOW_ERROR] cleanup limpeza falhou no balao {bubble.id}")
                continue

        self._save_debug_image("debug_after_cleanup.png", work_image)
        self._save_debug_image("after_cleanup.png", work_image)
        print("[FLOW] cleanup_done")
        print("[FLOW] limpeza concluida")
        self._save_debug_image("debug_before_render.png", work_image)
        print("[FLOW] render_start")
        for index, bubble in enumerate(editable_bubbles):
            self._check_timeout(deadline, f"render balao {bubble.id}")
            bubble_progress = index / total_editable
            self._progress(progress_callback, 0.80 + bubble_progress * 0.14, "inserindo traducao")
            render_start = perf_counter()
            record = flow_report[bubble.id]
            try:
                print(f"[RENDER] draw_after_cleanup page={page_label} balloon={bubble.id}")
                print(f"[RENDER] draw balloon={bubble.id} text=\"{bubble.translated_text}\"")
                work_image = self.renderer.render(work_image, bubble)
                record["render_ran"] = bool(getattr(bubble, "render_success", False))
                if not record["render_ran"] and not record["skipped_reason"]:
                    record["skipped_reason"] = "renderer nao desenhou a traducao"
                    print(f"[RENDER_SKIP] balloon={bubble.id} reason={record['skipped_reason']}")
                    print(f"[FLOW_ERROR] render renderer nao desenhou a traducao no balao {bubble.id}")
            except Exception as exc:
                if not record["skipped_reason"]:
                    record["skipped_reason"] = f"erro na renderizacao: {exc}"
                bubble.processing_notes.append(f"erro na renderizacao: {exc}")
                print(f"[RENDER_SKIP] balloon={bubble.id} reason={exc}")
                print(f"[FLOW_ERROR] render {exc}")
            render_elapsed += self._elapsed(render_start)
        drawn_count = sum(1 for bubble in editable_bubbles if bool(getattr(bubble, "render_success", False)))
        failed_count = max(0, len(editable_bubbles) - drawn_count)
        print(f"[RENDER_SUMMARY] drawn_count={drawn_count} failed_count={failed_count}")
        self._save_debug_image("debug_after_render.png", work_image)
        print("[FLOW] render_done")
        print("[FLOW] renderizacao concluida")

        timings["clean"] = round(clean_elapsed, 4)
        self._log_perf("clean", timings["clean"])
        timings["render"] = round(render_elapsed, 4)
        self._log_perf("render", timings["render"])

        if not bubbles:
            self._progress(progress_callback, 0.80, "nenhum balao detectado")

        self._progress(progress_callback, 0.96, "finalizando")
        # Skip debug image saving for mobile version
        # self._save_debug_image("debug_final_rendered.png", work_image)
        # self._save_debug_image("debug_final_translated.png", work_image)
        # self._save_debug_image("final.png", work_image)
        # self._save_debug_image("final_with_bbox.png", self._draw_debug_bboxes(work_image, bubbles))

        flow_report_list = self._finalize_flow_report(bubbles, flow_report)
        if flow_report_path is not None:
            self._save_flow_report(flow_report_path, flow_report_list)

        detected_count = len(bubbles)
        ocr_count = sum(1 for bubble in bubbles if bubble.source_text)
        translated_count = sum(1 for bubble in bubbles if bubble.translated_text)
        rendered_count = sum(1 for row in flow_report_list if row.get("render_ran"))
        cleanup_success_count = len(cleaned_bubble_ids)

        # Convert image to bytes for mobile (no disk save)
        step_start = perf_counter()
        import cv2
        success, buffer = cv2.imencode('.png', work_image)
        if not success:
            raise RuntimeError("Failed to encode image to PNG")
        image_bytes = buffer.tobytes()
        self._record_timing(timings, "encode", step_start)
        
        _ocr_count = sum(1 for b in bubbles if b.source_text)
        _tr_count = sum(1 for b in bubbles if b.translated_text)
        _rn_count = sum(1 for row in flow_report.values() if row.get("render_ran"))
        print(f"[FLOW] resumo: detectados={len(bubbles)} ocr={_ocr_count} traduzidos={_tr_count} renderizados={_rn_count}")
        self._raise_if_flow_incomplete(detected_count, ocr_count, translated_count, rendered_count, flow_report_list)
        self._record_timing(timings, "total", total_start)
        self._progress(progress_callback, 1.0, "finalizado")

        return AppResult(
            original_image_path=original_image_path,
            translated_image_path=None,
            bubbles=bubbles,
            output_dir=None,
            image_bytes=image_bytes,
            metadata={
                "bubble_count": len(bubbles),
                "translation_flow": safe_text(self.config.translation_flow) or "auto_to_en",
                "translation_provider": safe_text(translation_settings.provider) or "multilingual",
                "hf_translation_model": safe_text(translation_settings.model) or safe_text(getattr(self.config, "hf_translation_model", "")),
                "source_lang": safe_text(translation_settings.source_lang) or "auto",
                "ocr_lang": safe_text(getattr(self.config, "ocr_lang", "")) or "japan",
                "target_lang": safe_text(translation_settings.target_lang) or "en",
                "translation_style": safe_text(self.config.translation_style) or "natural",
                "performance_mode": safe_text(self.config.performance_mode) or "balanced",
                "timings": timings,
                "detected_count": detected_count,
                "ocr_count": ocr_count,
                "translated_count": translated_count,
                "rendered_count": rendered_count,
                "cleanup_success_count": cleaned_bubble_ids,
                "flow_success": bool(rendered_count > 0),
                "total_time": timings.get("total", 0.0),
                "bubble_flow_report_path": "",
                "cleaned_bubble_ids": sorted(cleaned_bubble_ids),
                "skipped_bubbles": [
                    {"id": bubble.id, "notes": list(bubble.processing_notes)}
                    for bubble in bubbles
                    if bubble.processing_notes
                ],
                "debug_dir": "",
            }
        )

    @staticmethod
    def _raise_if_flow_incomplete(
        detected_count: int,
        ocr_count: int,
        translated_count: int,
        rendered_count: int,
        flow_report_list: list[dict],
    ) -> None:
        if detected_count <= 0:
            message = "nenhum balao detectado; OCR/traducao/renderizacao nao executados"
            print(f"[FLOW_ERROR] detect {message}")
            raise RuntimeError(message)
        if ocr_count <= 0:
            reasons = MangaTranslatorPipeline._flow_error_reasons(flow_report_list)
            message = "OCR nao retornou texto em nenhum balao"
            if reasons:
                message = f"{message}: {reasons}"
            print(f"[FLOW_ERROR] ocr {message}")
            raise RuntimeError(message)
        if translated_count <= 0:
            reasons = MangaTranslatorPipeline._flow_error_reasons(flow_report_list)
            message = "traducao nao gerou texto valido em nenhum balao"
            if reasons:
                message = f"{message}: {reasons}"
            print(f"[FLOW_ERROR] translation {message}")
            raise RuntimeError(message)
        if rendered_count <= 0:
            reasons = MangaTranslatorPipeline._flow_error_reasons(flow_report_list)
            message = "renderizacao nao desenhou nenhuma traducao"
            if reasons:
                message = f"{message}: {reasons}"
            print(f"[FLOW_ERROR] render {message}")
            raise RuntimeError(message)

    @staticmethod
    def _flow_error_reasons(flow_report_list: list[dict]) -> str:
        reasons = []
        for row in flow_report_list:
            reason = safe_text(row.get("skipped_reason"))
            if reason:
                reasons.append(f"balao {row.get('id')}: {reason}")
        return "; ".join(reasons[:5])

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

    def _record_timing(self, timings: dict[str, float], stage: str, start: float) -> None:
        elapsed = self._elapsed(start)
        timings[stage] = elapsed
        self._log_perf(stage, elapsed)

    @staticmethod
    def _log_perf(stage: str, elapsed: float) -> None:
        print(f"[PERF] {stage}={elapsed:.4f}s")

    @staticmethod
    def _check_timeout(deadline: float, stage: str) -> None:
        if perf_counter() > deadline:
            message = f"timeout de processamento atingido em {stage}"
            print(f"[FLOW_ERROR] timeout {message}")
            raise TimeoutError(message)

    @staticmethod
    def _full_page_bubble(image: np.ndarray) -> SpeechBubble:
        height, width = image.shape[:2]
        mask = np.full((height, width), 255, dtype=np.uint8)
        bubble = SpeechBubble(id=1, bbox=BoundingBox(0, 0, width, height), mask=mask)
        bubble.processing_notes.append("fallback_pagina_inteira_sem_baloes_yolo")
        return bubble

    def _read_ocr_with_fallbacks(
        self,
        image: np.ndarray,
        bubble: SpeechBubble,
        page_label: str,
    ) -> tuple[list, list[str], str]:
        variants = self._ocr_crop_variants(image, bubble)
        if not variants:
            return [], [], "crop vazio para OCR"

        last_reason = "OCR vazio"
        for label, crop, offset_x, offset_y, input_scale in variants:
            print(f"[OCR_FALLBACK] page={page_label} balloon={bubble.id} strategy={label}")
            if crop is None or crop.size == 0:
                last_reason = f"{label}: crop vazio"
                print(f"[OCR_EMPTY] page={page_label} balloon={bubble.id} reason={last_reason}")
                continue
            height, width = crop.shape[:2]
            if width <= 1 or height <= 1:
                last_reason = f"{label}: crop invalido {width}x{height}"
                print(f"[OCR_EMPTY] page={page_label} balloon={bubble.id} reason={last_reason}")
                continue
            print(f"[OCR] crop_size page={page_label} balloon={bubble.id} size={width}x{height} strategy={label}")
            resized, scale = self._resize_crop_for_ocr(crop)
            if scale != 1.0:
                print(
                    f"[OCR] crop_size page={page_label} balloon={bubble.id} "
                    f"size={resized.shape[1]}x{resized.shape[0]} strategy={label}_scaled"
                )
            try:
                boxes = self.ocr.read(resized, offset_x=offset_x, offset_y=offset_y, force=True, use_cache=False)
            except Exception as exc:
                last_reason = f"{label}: erro no OCR: {exc}"
                print(f"[OCR_ERROR] page={page_label} balloon={bubble.id} error={exc}")
                continue
            total_scale = float(input_scale) * float(scale)
            if total_scale != 1.0:
                self._rescale_boxes(boxes, total_scale, offset_x, offset_y)
            texts = [safe_text(box.text) for box in boxes if safe_text(box.text)]
            if texts:
                if label != "masked":
                    bubble.processing_notes.append(f"OCR fallback aplicado: {label}")
                return boxes, texts, ""
            last_reason = f"{label}: OCR vazio"
            print(f"[OCR_EMPTY] page={page_label} balloon={bubble.id} reason={last_reason}")

        return [], [], last_reason

    def _ocr_crop_variants(self, image: np.ndarray, bubble: SpeechBubble) -> list[tuple[str, np.ndarray, int, int, float]]:
        image_h, image_w = image.shape[:2]
        raw_x1, raw_y1, raw_x2, raw_y2 = bubble.bbox.x1, bubble.bbox.y1, bubble.bbox.x2, bubble.bbox.y2
        x1 = max(0, min(image_w, int(raw_x1)))
        y1 = max(0, min(image_h, int(raw_y1)))
        x2 = max(0, min(image_w, int(raw_x2)))
        y2 = max(0, min(image_h, int(raw_y2)))
        if x2 <= x1 or y2 <= y1:
            return []
        variants: list[tuple[str, np.ndarray, int, int, float]] = []

        masked = self._masked_crop_for_ocr(image, bubble)
        variants.append(("masked", masked, x1, y1, 1.0))

        original = image[y1:y2, x1:x2]
        variants.append(("original", original, x1, y1, 1.0))

        margin = max(8, int(min(max(1, bubble.bbox.width), max(1, bubble.bbox.height)) * 0.12))
        mx1 = max(0, x1 - margin)
        my1 = max(0, y1 - margin)
        mx2 = min(image_w, x2 + margin)
        my2 = min(image_h, y2 + margin)
        variants.append(("margin", image[my1:my2, mx1:mx2], mx1, my1, 1.0))

        # Variantes 2x/3x removidas: _resize_crop_for_ocr ja faz upscale
        # adaptativo para crops pequenos; duplicar aqui multiplicava o pico
        # de RAM por balao em VPS pequena.

        contrast = self._contrast_ocr_crop(original)
        if contrast is not None and contrast.size > 0:
            variants.append(("contrast_gray", contrast, x1, y1, 1.0))

        return variants

    @staticmethod
    def _contrast_ocr_crop(crop: np.ndarray | None) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        contrast = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _rescale_boxes(boxes: list, scale: float, offset_x: int, offset_y: int) -> None:
        if scale <= 0 or scale == 1.0:
            return
        for box in boxes:
            scaled_polygon = []
            for x, y in box.polygon:
                local_x = (x - offset_x) / scale
                local_y = (y - offset_y) / scale
                scaled_polygon.append((int(round(offset_x + local_x)), int(round(offset_y + local_y))))
            box.polygon = scaled_polygon

    @staticmethod
    def _masked_crop_for_ocr(image: np.ndarray, bubble: SpeechBubble) -> np.ndarray:
        crop = image[bubble.bbox.y1 : bubble.bbox.y2, bubble.bbox.x1 : bubble.bbox.x2]
        if crop is None or crop.size == 0:
            return crop

        if getattr(bubble, "mask", None) is None or bubble.mask.size == 0:
            return crop

        mask_crop = bubble.mask[bubble.bbox.y1 : bubble.bbox.y2, bubble.bbox.x1 : bubble.bbox.x2]
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
    def _rescale_ocr_boxes(bubble: SpeechBubble, scale: float) -> None:
        if scale <= 0 or scale == 1.0:
            return
        ox = bubble.bbox.x1
        oy = bubble.bbox.y1
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
        if not bool(getattr(self.config, "debug_enabled", False)):
            return
        self._save_debug_image("debug_inner_mask.png", self.cleaner.last_inner_mask)
        self._save_debug_image("debug_dark_text_mask.png", self.cleaner.last_dark_mask)
        self._save_debug_image("debug_ocr_mask.png", self.cleaner.last_ocr_mask)
        self._save_debug_image("debug_ocr_text_mask.png", self.cleaner.last_ocr_mask)
        self._save_debug_image("debug_final_cleanup_mask.png", self.cleaner.last_cleanup_mask)

        self._save_debug_image(f"bubble_{bubble.id}_inner_mask.png", self.cleaner.last_inner_mask)
        self._save_debug_image(f"bubble_{bubble.id}_dark_text_mask.png", self.cleaner.last_dark_mask)
        self._save_debug_image(f"bubble_{bubble.id}_ocr_mask.png", self.cleaner.last_ocr_mask)
        self._save_debug_image(f"bubble_{bubble.id}_ocr_text_mask.png", self.cleaner.last_ocr_mask)
        self._save_debug_image(f"bubble_{bubble.id}_final_cleanup_mask.png", self.cleaner.last_cleanup_mask)

    def _save_debug_image(self, filename: str, image: np.ndarray | None) -> None:
        if (
            self.debug_dir is None
            or image is None
            or image.size == 0
            or not bool(getattr(self.config, "debug_enabled", False))
        ):
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


def process_image(
    image_path: Path | str,
    output_dir: Path | str | None = None,
    config: AppConfig | None = None,
    progress_callback=None,
) -> AppResult:
    """Compatibility wrapper for callers that import process_image directly."""
    pipeline = MangaTranslatorPipeline(config=config)
    return pipeline.run(image_path, output_dir=output_dir, progress_callback=progress_callback)
