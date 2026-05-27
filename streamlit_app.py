from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import time
from uuid import uuid4
import zipfile

import streamlit as st

from app.backend import process_uploaded_image
from app.pdf_utils import get_pdf_page_count, iter_pdf_pages
from app.utils import ensure_dir, get_translation_flow_config, get_translation_flow_labels


translation_style = "natural"
translation_style_label = "Natural"
debug_enabled = False


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def file_hash(content: bytes) -> str:
    digest = sha256()
    digest.update(content)
    return digest.hexdigest()


def page_job_key(
    file_hash_value: str,
    file_name: str,
    page_number: int,
    translation_flow: str,
    source_lang: str,
    target_lang: str,
    ocr_lang: str,
    ocr_engine: str,
    performance_mode: str,
    translation_style: str,
    translation_provider: str,
) -> str:
    digest = sha256()
    digest.update(str(file_hash_value).encode("utf-8", errors="ignore"))
    digest.update(clean_text(file_name).encode("utf-8", errors="ignore"))
    digest.update(f"_page_{page_number}_".encode("utf-8"))
    digest.update(clean_text(translation_flow).encode("utf-8", errors="ignore"))
    digest.update(clean_text(source_lang).encode("utf-8", errors="ignore"))
    digest.update(clean_text(target_lang).encode("utf-8", errors="ignore"))
    digest.update(clean_text(ocr_lang).encode("utf-8", errors="ignore"))
    digest.update(clean_text(ocr_engine).encode("utf-8", errors="ignore"))
    digest.update(clean_text(performance_mode).encode("utf-8", errors="ignore"))
    digest.update(clean_text(translation_style).encode("utf-8", errors="ignore"))
    digest.update(clean_text(translation_provider).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def init_state() -> None:
    if "translated_pages" not in st.session_state:
        st.session_state["translated_pages"] = {}
    if "translation_errors" not in st.session_state:
        st.session_state["translation_errors"] = {}
    if "realtime_jobs" not in st.session_state:
        st.session_state["realtime_jobs"] = {}
    if "realtime_active_jobs" not in st.session_state:
        st.session_state["realtime_active_jobs"] = set()


def create_realtime_job(total_pages: int) -> str:
    job_id = uuid4().hex[:12]
    st.session_state["realtime_jobs"][job_id] = {
        "job_id": job_id,
        "created_at": time(),
        "total_pages": int(total_pages),
        "status": "pending",
        "current_page": 0,
        "pages": {idx: {"page": idx, "status": "pending", "error": ""} for idx in range(1, int(total_pages) + 1)},
        "error": "",
    }
    print(f"[REALTIME] job_created job_id={job_id}")
    return job_id


def set_job_status(job_id: str, status: str, page: int = 0, error: str = "") -> None:
    jobs = st.session_state.get("realtime_jobs", {})
    job = jobs.get(job_id)
    if not job:
        return
    job["status"] = clean_text(status) or job["status"]
    if page > 0:
        job["current_page"] = int(page)
    if error:
        job["error"] = clean_text(error)


def set_page_status(job_id: str, page: int, status: str, error: str = "") -> None:
    jobs = st.session_state.get("realtime_jobs", {})
    job = jobs.get(job_id)
    if not job:
        return
    page_key = int(page)
    page_state = job["pages"].setdefault(
        page_key,
        {
            "page": page_key,
            "status": "pending",
            "error": "",
        },
    )
    page_state["status"] = clean_text(status) or page_state["status"]
    if error:
        page_state["error"] = clean_text(error)
    job["current_page"] = max(int(job.get("current_page") or 0), page_key)


def render_realtime_panel(container, job_id: str) -> None:
    job = st.session_state.get("realtime_jobs", {}).get(job_id) or {}
    total_pages = int(job.get("total_pages") or 0)
    current_page = int(job.get("current_page") or 0)
    status = clean_text(job.get("status")) or "pending"
    pages = job.get("pages") or {}

    with container.container():
        st.markdown("## Progresso em tempo real")
        st.caption(f"job_id={job_id}")
        st.text(f"Status: {status} | Pagina atual: {current_page}/{total_pages}")
        ordered_pages = [pages[key] for key in sorted(pages.keys())]
        if ordered_pages:
            st.table(ordered_pages)


def normalize_cache_entry(value):
    if value is None:
        return None
    if isinstance(value, dict):
        translated_path = clean_text(value.get("translated_path"))
        if translated_path:
            return {"translated_path": translated_path, "timings": value.get("timings") or {}}
    translated_path = getattr(value, "translated_image_path", None)
    if translated_path:
        return {"translated_path": str(translated_path), "timings": getattr(value, "metadata", {}).get("timings", {})}
    return None


def format_timings(timings) -> str:
    if not isinstance(timings, dict):
        return ""
    parts = []
    for key in ("detect", "ocr", "translate", "clean", "render", "total"):
        value = timings.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.1f}s")
    return " | ".join(parts)


def validate_pipeline_result(result) -> None:
    metadata = getattr(result, "metadata", {}) or {}
    translated_count = int(metadata.get("translated_count") or 0)
    rendered_count = int(metadata.get("rendered_count") or 0)
    detected_count = int(metadata.get("detected_count") or metadata.get("bubble_count") or 0)
    if translated_count > 0 and rendered_count <= 0:
        skipped = metadata.get("skipped_bubbles") or []
        details = "; ".join(clean_text(item.get("notes")) for item in skipped if isinstance(item, dict))
        message = "Texto traduzido mas limpeza/renderizacao falhou em todos os baloes."
        if details:
            message = f"{message} Motivos: {details}"
        raise RuntimeError(message)
    elif detected_count > 0 and translated_count <= 0:
        raise RuntimeError("Baloes detectados, mas nenhuma traducao valida foi gerada para renderizar.")


def render_page_success(slot, page_number: int, translated_path: Path, job_key: str) -> None:
    with slot.container():
        st.markdown(f"### Pagina {page_number}")
        st.image(str(translated_path), use_container_width=True)
        with open(translated_path, "rb") as image_file:
            st.download_button(
                "Baixar pagina traduzida",
                data=image_file,
                file_name=translated_path.name,
                mime="image/png",
                key=f"download_{job_key}",
                use_container_width=True,
            )
        st.divider()


def render_page_error(slot, page_number: int, message: str) -> None:
    with slot.container():
        st.markdown(f"### Pagina {page_number}")
        st.error(f"Nao foi possivel traduzir a pagina {page_number}.")
        if clean_text(message):
            with st.expander("Detalhes tecnicos"):
                st.code(clean_text(message), language="text")
        st.divider()


def build_zip_bytes(items: list[tuple[int, Path]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for page_number, translated_path in items:
            arcname = f"page_{int(page_number):04d}.png"
            archive.writestr(arcname, translated_path.read_bytes())
    buffer.seek(0)
    return buffer.getvalue()


def gather_inputs(uploaded_files, status_placeholder):
    files_data = []
    total_pages = 0

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        raw_name = clean_text(uploaded_file.name) or f"arquivo_{idx}"
        suffix = Path(raw_name).suffix.lower()
        content = uploaded_file.getvalue()
        hash_value = file_hash(content)

        if suffix == ".pdf":
            try:
                count = get_pdf_page_count(content)
            except Exception as exc:
                count = 1
                files_data.append(
                    {
                        "kind": "pdf_error",
                        "name": raw_name,
                        "content": content,
                        "hash": hash_value,
                        "page_count": count,
                        "error": clean_text(exc) or "Falha ao abrir PDF.",
                    }
                )
                total_pages += count
                continue

            files_data.append(
                {
                    "kind": "pdf",
                    "name": raw_name,
                    "content": content,
                    "hash": hash_value,
                    "page_count": max(0, int(count)),
                    "error": "",
                }
            )
            total_pages += max(0, int(count))
            continue

        files_data.append(
            {
                "kind": "image",
                "name": raw_name,
                "content": content,
                "hash": hash_value,
                "page_count": 1,
                "error": "",
            }
        )
        total_pages += 1

    if total_pages <= 0:
        status_placeholder.warning("Nenhuma pagina valida encontrada.")
    return files_data, total_pages


st.set_page_config(page_title="Manga Translator", page_icon="📚", layout="centered")
init_state()

st.title("Manga Translator")
st.caption("Upload de imagens ou PDF. Traducao automatica pagina por pagina em modo vertical.")
st.caption("Motor de traducao: usa traducao leve por padrao para nao sobrecarregar VPS pequena.")

mode_labels = get_translation_flow_labels()
label_to_mode = {label: mode for mode, label in mode_labels.items()}
selected_mode_label = st.selectbox("Modo de traducao", list(label_to_mode.keys()), index=0)
selected_mode = label_to_mode[selected_mode_label]
selected_mode_config = get_translation_flow_config(selected_mode)
source_lang = selected_mode_config["source_lang"]
target_lang = selected_mode_config["target_lang"]
ocr_lang = selected_mode_config["ocr_lang"]
translation_flow = selected_mode_config["mode"]
translation_flow_label = selected_mode_config["label"]

ocr_engine_labels = {
    "Automático": "auto",
    "PaddleOCR": "paddle",
    "Manga OCR": "manga",
}
selected_ocr_engine_label = st.selectbox("Motor de OCR", list(ocr_engine_labels.keys()), index=0)
ocr_engine = ocr_engine_labels[selected_ocr_engine_label]
st.caption("Manga OCR é recomendado para texto japonês de mangá.")

performance_labels = {
    "Rapido": "fast",
    "Equilibrado": "balanced",
    "Qualidade": "quality",
}
selected_performance_label = st.selectbox("Desempenho", list(performance_labels.keys()), index=0)
performance_mode = performance_labels[selected_performance_label]
performance_label = selected_performance_label
st.caption(
    "Tradução multilíngue local. Configure via env: "
    "TRANSLATION_PROVIDER, TRANSLATION_MODEL, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG."
)

import os as _os

translation_provider = _os.getenv("TRANSLATION_PROVIDER", "multilingual")

uploaded_files = st.file_uploader(
    "Envie imagens ou PDF de manga",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Envie um PDF ou imagens para iniciar.")
    st.stop()

status_placeholder = st.empty()
overall_progress = st.progress(0)

st.markdown("## Paginas traduzidas")

status_placeholder.info("Lendo arquivos enviados...")
inputs, total_pages = gather_inputs(uploaded_files, status_placeholder)
if total_pages <= 0:
    st.stop()
print("[REALTIME] upload_received")
job_id = create_realtime_job(total_pages)
if job_id in st.session_state["realtime_active_jobs"]:
    status_placeholder.warning("Job ja em processamento. Aguarde terminar.")
    st.stop()
st.session_state["realtime_active_jobs"].add(job_id)
set_job_status(job_id, "loading")
print(f"[REALTIME] processing_started job_id={job_id}")
realtime_panel = st.empty()
render_realtime_panel(realtime_panel, job_id)

output_pdf_pages_dir = ensure_dir(Path("output") / "_pdf_pages")
try:
    _pdf_dpi_fast = int(_os.getenv("PDF_DPI_FAST", "110") or "110")
    _pdf_dpi_default = int(_os.getenv("PDF_DPI_DEFAULT", "130") or "130")
except ValueError:
    _pdf_dpi_fast, _pdf_dpi_default = 110, 130
pdf_dpi = _pdf_dpi_fast if performance_mode == "fast" else _pdf_dpi_default

translated_for_zip: list[tuple[int, Path]] = []
global_page_counter = 0
finished_pages = 0

for file_idx, file_item in enumerate(inputs, start=1):
    file_kind = file_item["kind"]
    file_name = file_item["name"]
    file_content = file_item["content"]
    hash_value = file_item["hash"]

    if file_kind == "pdf_error":
        global_page_counter += 1
        finished_pages += 1
        set_page_status(job_id, global_page_counter, "failed", file_item["error"])
        print(f"[REALTIME_ERROR] job_id={job_id} page={global_page_counter} stage=loading error={clean_text(file_item['error'])}")
        render_realtime_panel(realtime_panel, job_id)
        slot = st.empty()
        job_key = page_job_key(
            hash_value,
            file_name,
            1,
            translation_flow,
            source_lang,
            target_lang,
            ocr_lang,
            ocr_engine,
            performance_mode,
            translation_style,
            translation_provider,
        )
        st.session_state["translation_errors"][job_key] = {"message": file_item["error"]}
        status_placeholder.warning(f"Nao foi possivel abrir o PDF {file_name}.")
        render_page_error(slot, global_page_counter, file_item["error"])
        overall_progress.progress(int((finished_pages / total_pages) * 100))
        continue

    if file_kind == "image":
        page_number_in_file = 1
        global_page_counter += 1
        set_page_status(job_id, global_page_counter, "loading")
        set_job_status(job_id, "loading", page=global_page_counter)
        print(f"[REALTIME] page_loaded job_id={job_id} page={global_page_counter}")
        render_realtime_panel(realtime_panel, job_id)
        job_key = page_job_key(
            hash_value,
            file_name,
            page_number_in_file,
            translation_flow,
            source_lang,
            target_lang,
            ocr_lang,
            ocr_engine,
            performance_mode,
            translation_style,
            translation_provider,
        )
        slot = st.empty()

        cached_result = normalize_cache_entry(st.session_state["translated_pages"].get(job_key))
        cached_error = st.session_state["translation_errors"].get(job_key)

        if cached_result is not None:
            translated_path = Path(cached_result["translated_path"])
            if translated_path.exists():
                render_page_success(slot, global_page_counter, translated_path, job_key)
                translated_for_zip.append((global_page_counter, translated_path))
                timing_summary = format_timings(cached_result.get("timings"))
                status_placeholder.info(
                    f"Pagina {global_page_counter} reutilizada do cache"
                    + (f" ({timing_summary})" if timing_summary else ".")
                )
            else:
                st.session_state["translated_pages"].pop(job_key, None)
                st.session_state["translation_errors"][job_key] = {"message": "Arquivo traduzido nao encontrado no cache."}
                render_page_error(slot, global_page_counter, "Arquivo traduzido nao encontrado no cache.")
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            continue

        if cached_error is not None:
            render_page_error(slot, global_page_counter, clean_text(cached_error.get("message")))
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            continue

        status_placeholder.info(
            f"Usando Manga OCR... Traduzindo pagina {global_page_counter} de {total_pages} · {translation_flow_label}..."
            if ocr_engine == "manga" or (ocr_engine == "auto" and ocr_lang in {"japan", "ja", "japanese"})
            else f"Traduzindo pagina {global_page_counter} de {total_pages} · {translation_flow_label}..."
        )

        def update_progress(value: float, message: str, meta=None) -> None:
            stage_id = clean_text((meta or {}).get("stage_id")) if isinstance(meta, dict) else ""
            stage_map = {
                "load_image": "loading",
                "detect": "detecting_balloons",
                "ocr": "ocr",
                "translate": "translating",
                "clean": "cleaning",
                "render": "rendering",
                "finish": "done",
            }
            mapped_status = stage_map.get(stage_id, "")
            if mapped_status:
                set_page_status(job_id, global_page_counter, mapped_status)
                set_job_status(job_id, mapped_status, page=global_page_counter)
                if mapped_status == "detecting_balloons":
                    print(f"[REALTIME] detecting_balloons job_id={job_id} page={global_page_counter}")
                elif mapped_status == "ocr":
                    print(f"[REALTIME] ocr_start job_id={job_id} page={global_page_counter}")
                elif mapped_status == "translating":
                    print(f"[REALTIME] translation_start job_id={job_id} page={global_page_counter}")
                elif mapped_status == "cleaning":
                    print(f"[REALTIME] cleaning_start job_id={job_id} page={global_page_counter}")
                elif mapped_status == "rendering":
                    print(f"[REALTIME] rendering_start job_id={job_id} page={global_page_counter}")
                render_realtime_panel(realtime_panel, job_id)
            del message
            local_value = float(max(0.0, min(1.0, value)))
            absolute_progress = ((finished_pages + local_value) / total_pages) * 100.0
            overall_progress.progress(int(max(0, min(100, round(absolute_progress)))))

        try:
            result = process_uploaded_image(
                filename=file_name,
                content=file_content,
                translation_flow=translation_flow,
                source_lang=source_lang,
                target_lang=target_lang,
                ocr_lang=ocr_lang,
                ocr_engine=ocr_engine,
                translation_style=translation_style,
                performance_mode=performance_mode,
                translation_provider=translation_provider,
                debug_enabled=debug_enabled,
                progress_callback=update_progress,
            )
            validate_pipeline_result(result)
            set_page_status(job_id, global_page_counter, "done")
            print(f"[REALTIME] ocr_done job_id={job_id} page={global_page_counter}")
            print(f"[REALTIME] translation_done job_id={job_id} page={global_page_counter}")
            print(f"[REALTIME] cleaning_done job_id={job_id} page={global_page_counter}")
            print(f"[REALTIME] rendering_done job_id={job_id} page={global_page_counter}")
            translated_path = Path(result.translated_image_path)
            timings = result.metadata.get("timings", {}) if isinstance(result.metadata, dict) else {}
            st.session_state["translated_pages"][job_key] = {
                "translated_path": str(translated_path),
                "timings": timings,
            }
            st.session_state["translation_errors"].pop(job_key, None)
            render_page_success(slot, global_page_counter, translated_path, job_key)
            translated_for_zip.append((global_page_counter, translated_path))
            timing_summary = format_timings(timings)
            status_placeholder.success(
                f"Pagina {global_page_counter} concluida. ({translation_flow_label} · {performance_label})"
                + (f" {timing_summary}" if timing_summary else "")
            )
            print(f"[REALTIME] page_done job_id={job_id} page={global_page_counter}")
            render_realtime_panel(realtime_panel, job_id)
        except Exception as exc:
            message = clean_text(exc) or f"Falha na pagina {global_page_counter}."
            set_page_status(job_id, global_page_counter, "failed", message)
            st.session_state["translation_errors"][job_key] = {"message": message}
            render_page_error(slot, global_page_counter, message)
            status_placeholder.warning(f"Nao foi possivel traduzir a pagina {global_page_counter}.")
            print(f"[REALTIME_ERROR] job_id={job_id} page={global_page_counter} stage=processing error={message}")
            render_realtime_panel(realtime_panel, job_id)
        finally:
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))
        continue

    # PDF
    temp_pdf_dir = ensure_dir(output_pdf_pages_dir / f"pdf_{file_idx}_{hash_value}")
    status_placeholder.info(f"Lendo PDF: {file_name}")

    for page_number_in_file, total_pdf_pages, page_path in iter_pdf_pages(file_content, temp_pdf_dir, dpi=pdf_dpi):
        global_page_counter += 1
        set_page_status(job_id, global_page_counter, "loading")
        set_job_status(job_id, "loading", page=global_page_counter)
        print(f"[REALTIME] page_loaded job_id={job_id} page={global_page_counter}")
        render_realtime_panel(realtime_panel, job_id)
        job_key = page_job_key(
            hash_value,
            file_name,
            page_number_in_file,
            translation_flow,
            source_lang,
            target_lang,
            ocr_lang,
            ocr_engine,
            performance_mode,
            translation_style,
            translation_provider,
        )
        slot = st.empty()

        cached_result = normalize_cache_entry(st.session_state["translated_pages"].get(job_key))
        cached_error = st.session_state["translation_errors"].get(job_key)

        if cached_result is not None:
            translated_path = Path(cached_result["translated_path"])
            if translated_path.exists():
                render_page_success(slot, global_page_counter, translated_path, job_key)
                translated_for_zip.append((global_page_counter, translated_path))
                timing_summary = format_timings(cached_result.get("timings"))
                status_placeholder.info(
                    f"Pagina {global_page_counter} reutilizada do cache"
                    + (f" ({timing_summary})" if timing_summary else ".")
                )
            else:
                st.session_state["translated_pages"].pop(job_key, None)
                st.session_state["translation_errors"][job_key] = {"message": "Arquivo traduzido nao encontrado no cache."}
                render_page_error(slot, global_page_counter, "Arquivo traduzido nao encontrado no cache.")
            finished_pages += 1
            set_page_status(job_id, global_page_counter, "done")
            print(f"[REALTIME] page_done job_id={job_id} page={global_page_counter}")
            render_realtime_panel(realtime_panel, job_id)
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            status_placeholder.success(f"Pagina {global_page_counter} concluida. ({translation_flow_label})")
            continue

        if cached_error is not None:
            render_page_error(slot, global_page_counter, clean_text(cached_error.get("message")))
            finished_pages += 1
            set_page_status(job_id, global_page_counter, "failed", clean_text(cached_error.get("message")))
            print(f"[REALTIME_ERROR] job_id={job_id} page={global_page_counter} stage=cache error={clean_text(cached_error.get('message'))}")
            render_realtime_panel(realtime_panel, job_id)
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            status_placeholder.warning(f"Nao foi possivel traduzir a pagina {global_page_counter}.")
            continue

        status_placeholder.info(f"Convertendo pagina {global_page_counter} de {total_pages}...")
        page_bytes = b""
        try:
            page_bytes = page_path.read_bytes()
        except Exception as exc:
            message = clean_text(exc) or f"Falha ao ler pagina {global_page_counter}."
            set_page_status(job_id, global_page_counter, "failed", message)
            st.session_state["translation_errors"][job_key] = {"message": message}
            render_page_error(slot, global_page_counter, message)
            finished_pages += 1
            print(f"[REALTIME_ERROR] job_id={job_id} page={global_page_counter} stage=loading error={message}")
            render_realtime_panel(realtime_panel, job_id)
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            continue

        status_placeholder.info(
            f"Usando Manga OCR... Traduzindo pagina {global_page_counter} de {total_pages} · {translation_flow_label}..."
            if ocr_engine == "manga" or (ocr_engine == "auto" and ocr_lang in {"japan", "ja", "japanese"})
            else f"Traduzindo pagina {global_page_counter} de {total_pages} · {translation_flow_label}..."
        )

        def update_progress(value: float, message: str, meta=None) -> None:
            stage_id = clean_text((meta or {}).get("stage_id")) if isinstance(meta, dict) else ""
            stage_map = {
                "load_image": "loading",
                "detect": "detecting_balloons",
                "ocr": "ocr",
                "translate": "translating",
                "clean": "cleaning",
                "render": "rendering",
                "finish": "done",
            }
            mapped_status = stage_map.get(stage_id, "")
            if mapped_status:
                set_page_status(job_id, global_page_counter, mapped_status)
                set_job_status(job_id, mapped_status, page=global_page_counter)
                if mapped_status == "detecting_balloons":
                    print(f"[REALTIME] detecting_balloons job_id={job_id} page={global_page_counter}")
                elif mapped_status == "ocr":
                    print(f"[REALTIME] ocr_start job_id={job_id} page={global_page_counter}")
                elif mapped_status == "translating":
                    print(f"[REALTIME] translation_start job_id={job_id} page={global_page_counter}")
                elif mapped_status == "cleaning":
                    print(f"[REALTIME] cleaning_start job_id={job_id} page={global_page_counter}")
                elif mapped_status == "rendering":
                    print(f"[REALTIME] rendering_start job_id={job_id} page={global_page_counter}")
                render_realtime_panel(realtime_panel, job_id)
            del message
            local_value = float(max(0.0, min(1.0, value)))
            absolute_progress = ((finished_pages + local_value) / total_pages) * 100.0
            overall_progress.progress(int(max(0, min(100, round(absolute_progress)))))

        try:
            result = process_uploaded_image(
                filename=f"{Path(file_name).stem}_page_{page_number_in_file:04d}.png",
                content=page_bytes,
                translation_flow=translation_flow,
                source_lang=source_lang,
                target_lang=target_lang,
                ocr_lang=ocr_lang,
                ocr_engine=ocr_engine,
                translation_style=translation_style,
                performance_mode=performance_mode,
                translation_provider=translation_provider,
                debug_enabled=debug_enabled,
                progress_callback=update_progress,
            )
            validate_pipeline_result(result)
            set_page_status(job_id, global_page_counter, "done")
            print(f"[REALTIME] ocr_done job_id={job_id} page={global_page_counter}")
            print(f"[REALTIME] translation_done job_id={job_id} page={global_page_counter}")
            print(f"[REALTIME] cleaning_done job_id={job_id} page={global_page_counter}")
            print(f"[REALTIME] rendering_done job_id={job_id} page={global_page_counter}")
            translated_path = Path(result.translated_image_path)
            timings = result.metadata.get("timings", {}) if isinstance(result.metadata, dict) else {}
            st.session_state["translated_pages"][job_key] = {
                "translated_path": str(translated_path),
                "timings": timings,
            }
            st.session_state["translation_errors"].pop(job_key, None)
            render_page_success(slot, global_page_counter, translated_path, job_key)
            translated_for_zip.append((global_page_counter, translated_path))
            timing_summary = format_timings(timings)
            status_placeholder.success(
                f"Pagina {global_page_counter} concluida. ({translation_flow_label} · {performance_label})"
                + (f" {timing_summary}" if timing_summary else "")
            )
            print(f"[REALTIME] page_done job_id={job_id} page={global_page_counter}")
            render_realtime_panel(realtime_panel, job_id)
        except Exception as exc:
            message = clean_text(exc) or f"Falha na pagina {global_page_counter}."
            set_page_status(job_id, global_page_counter, "failed", message)
            st.session_state["translation_errors"][job_key] = {"message": message}
            render_page_error(slot, global_page_counter, message)
            status_placeholder.warning(f"Nao foi possivel traduzir a pagina {global_page_counter}.")
            print(f"[REALTIME_ERROR] job_id={job_id} page={global_page_counter} stage=processing error={message}")
            render_realtime_panel(realtime_panel, job_id)
        finally:
            page_bytes = b""
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))

if not translated_for_zip:
    status_placeholder.error("Finalizado com erro: nenhuma pagina traduzida.")
    set_job_status(job_id, "failed", error="nenhuma pagina traduzida")
elif len(translated_for_zip) < total_pages:
    status_placeholder.warning(f"Finalizado com erros: {len(translated_for_zip)} de {total_pages} paginas traduzidas.")
    set_job_status(job_id, "failed", error="finalizado com erros")
else:
    status_placeholder.success("Finalizado.")
    set_job_status(job_id, "done")
print(f"[REALTIME] job_done job_id={job_id}")
st.session_state["realtime_active_jobs"].discard(job_id)
render_realtime_panel(realtime_panel, job_id)

if translated_for_zip:
    ordered = sorted(translated_for_zip, key=lambda item: item[0])
    zip_data = build_zip_bytes(ordered)
    st.download_button(
        "Baixar todas as paginas traduzidas (.zip)",
        data=zip_data,
        file_name="paginas_traduzidas.zip",
        mime="application/zip",
        key="download_all_pages_zip",
        use_container_width=True,
    )
