from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import streamlit as st

from app.backend import process_uploaded_image
from app.pdf_utils import get_pdf_page_count, iter_pdf_pages
from app.utils import ensure_dir, get_translation_mode_config, get_translation_mode_labels


translation_style = "natural"
translation_style_label = "Natural"
performance_mode = "balanced"
performance_label = "Equilibrado"
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
    translation_mode: str,
    source_lang: str,
    target_lang: str,
    ocr_lang: str,
) -> str:
    digest = sha256()
    digest.update(str(file_hash_value).encode("utf-8", errors="ignore"))
    digest.update(clean_text(file_name).encode("utf-8", errors="ignore"))
    digest.update(f"_page_{page_number}_".encode("utf-8"))
    digest.update(clean_text(translation_mode).encode("utf-8", errors="ignore"))
    digest.update(clean_text(source_lang).encode("utf-8", errors="ignore"))
    digest.update(clean_text(target_lang).encode("utf-8", errors="ignore"))
    digest.update(clean_text(ocr_lang).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def init_state() -> None:
    if "translated_pages" not in st.session_state:
        st.session_state["translated_pages"] = {}
    if "translation_errors" not in st.session_state:
        st.session_state["translation_errors"] = {}


def normalize_cache_entry(value):
    if value is None:
        return None
    if isinstance(value, dict):
        translated_path = clean_text(value.get("translated_path"))
        if translated_path:
            return {"translated_path": translated_path}
    translated_path = getattr(value, "translated_image_path", None)
    if translated_path:
        return {"translated_path": str(translated_path)}
    return None


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

mode_labels = get_translation_mode_labels()
label_to_mode = {label: mode for mode, label in mode_labels.items()}
selected_mode_label = st.selectbox("Modo de traducao", list(label_to_mode.keys()), index=0)
selected_mode = label_to_mode[selected_mode_label]
selected_mode_config = get_translation_mode_config(selected_mode)
source_lang = selected_mode_config["source_lang"]
target_lang = selected_mode_config["target_lang"]
ocr_lang = selected_mode_config["ocr_lang"]
translation_mode = selected_mode_config["mode"]
translation_mode_label = selected_mode_config["label"]

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

output_pdf_pages_dir = ensure_dir(Path("output") / "_pdf_pages")

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
        slot = st.empty()
        job_key = page_job_key(hash_value, file_name, 1, translation_mode, source_lang, target_lang, ocr_lang)
        st.session_state["translation_errors"][job_key] = {"message": file_item["error"]}
        status_placeholder.warning(f"Nao foi possivel abrir o PDF {file_name}.")
        render_page_error(slot, global_page_counter, file_item["error"])
        overall_progress.progress(int((finished_pages / total_pages) * 100))
        continue

    if file_kind == "image":
        page_number_in_file = 1
        global_page_counter += 1
        job_key = page_job_key(
            hash_value,
            file_name,
            page_number_in_file,
            translation_mode,
            source_lang,
            target_lang,
            ocr_lang,
        )
        slot = st.empty()

        cached_result = normalize_cache_entry(st.session_state["translated_pages"].get(job_key))
        cached_error = st.session_state["translation_errors"].get(job_key)

        if cached_result is not None:
            translated_path = Path(cached_result["translated_path"])
            if translated_path.exists():
                render_page_success(slot, global_page_counter, translated_path, job_key)
                translated_for_zip.append((global_page_counter, translated_path))
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
            f"Traduzindo pagina {global_page_counter} de {total_pages} · {translation_mode_label}..."
        )

        def update_progress(value: float, message: str, meta=None) -> None:
            del message
            del meta
            local_value = float(max(0.0, min(1.0, value)))
            absolute_progress = ((finished_pages + local_value) / total_pages) * 100.0
            overall_progress.progress(int(max(0, min(100, round(absolute_progress)))))

        try:
            result = process_uploaded_image(
                filename=file_name,
                content=file_content,
                translation_mode=translation_mode,
                source_lang=source_lang,
                target_lang=target_lang,
                ocr_lang=ocr_lang,
                translation_style=translation_style,
                performance_mode=performance_mode,
                debug_enabled=debug_enabled,
                progress_callback=update_progress,
            )
            translated_path = Path(result.translated_image_path)
            st.session_state["translated_pages"][job_key] = {"translated_path": str(translated_path)}
            st.session_state["translation_errors"].pop(job_key, None)
            render_page_success(slot, global_page_counter, translated_path, job_key)
            translated_for_zip.append((global_page_counter, translated_path))
            status_placeholder.success(f"Pagina {global_page_counter} concluida. ({translation_mode_label})")
        except Exception as exc:
            message = clean_text(exc) or f"Falha na pagina {global_page_counter}."
            st.session_state["translation_errors"][job_key] = {"message": message}
            render_page_error(slot, global_page_counter, message)
            status_placeholder.warning(f"Nao foi possivel traduzir a pagina {global_page_counter}.")
        finally:
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))
        continue

    # PDF
    temp_pdf_dir = ensure_dir(output_pdf_pages_dir / f"pdf_{file_idx}_{hash_value}")
    status_placeholder.info(f"Lendo PDF: {file_name}")

    for page_number_in_file, total_pdf_pages, page_path in iter_pdf_pages(file_content, temp_pdf_dir, dpi=220):
        global_page_counter += 1
        job_key = page_job_key(
            hash_value,
            file_name,
            page_number_in_file,
            translation_mode,
            source_lang,
            target_lang,
            ocr_lang,
        )
        slot = st.empty()

        cached_result = normalize_cache_entry(st.session_state["translated_pages"].get(job_key))
        cached_error = st.session_state["translation_errors"].get(job_key)

        if cached_result is not None:
            translated_path = Path(cached_result["translated_path"])
            if translated_path.exists():
                render_page_success(slot, global_page_counter, translated_path, job_key)
                translated_for_zip.append((global_page_counter, translated_path))
            else:
                st.session_state["translated_pages"].pop(job_key, None)
                st.session_state["translation_errors"][job_key] = {"message": "Arquivo traduzido nao encontrado no cache."}
                render_page_error(slot, global_page_counter, "Arquivo traduzido nao encontrado no cache.")
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            status_placeholder.success(f"Pagina {global_page_counter} concluida. ({translation_mode_label})")
            continue

        if cached_error is not None:
            render_page_error(slot, global_page_counter, clean_text(cached_error.get("message")))
            finished_pages += 1
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
            st.session_state["translation_errors"][job_key] = {"message": message}
            render_page_error(slot, global_page_counter, message)
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            continue

        status_placeholder.info(
            f"Traduzindo pagina {global_page_counter} de {total_pages} · {translation_mode_label}..."
        )

        def update_progress(value: float, message: str, meta=None) -> None:
            del message
            del meta
            local_value = float(max(0.0, min(1.0, value)))
            absolute_progress = ((finished_pages + local_value) / total_pages) * 100.0
            overall_progress.progress(int(max(0, min(100, round(absolute_progress)))))

        try:
            result = process_uploaded_image(
                filename=f"{Path(file_name).stem}_page_{page_number_in_file:04d}.png",
                content=page_bytes,
                translation_mode=translation_mode,
                source_lang=source_lang,
                target_lang=target_lang,
                ocr_lang=ocr_lang,
                translation_style=translation_style,
                performance_mode=performance_mode,
                debug_enabled=debug_enabled,
                progress_callback=update_progress,
            )
            translated_path = Path(result.translated_image_path)
            st.session_state["translated_pages"][job_key] = {"translated_path": str(translated_path)}
            st.session_state["translation_errors"].pop(job_key, None)
            render_page_success(slot, global_page_counter, translated_path, job_key)
            translated_for_zip.append((global_page_counter, translated_path))
            status_placeholder.success(f"Pagina {global_page_counter} concluida. ({translation_mode_label})")
        except Exception as exc:
            message = clean_text(exc) or f"Falha na pagina {global_page_counter}."
            st.session_state["translation_errors"][job_key] = {"message": message}
            render_page_error(slot, global_page_counter, message)
            status_placeholder.warning(f"Nao foi possivel traduzir a pagina {global_page_counter}.")
        finally:
            page_bytes = b""
            if page_path.exists():
                page_path.unlink(missing_ok=True)
            finished_pages += 1
            overall_progress.progress(int((finished_pages / total_pages) * 100))

status_placeholder.success("Finalizado.")

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
