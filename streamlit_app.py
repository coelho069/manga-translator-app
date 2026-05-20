from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import streamlit as st

from app.auth import require_auth
from app.backend import create_batch_zip, process_batch, process_uploaded_image
from app.pdf_utils import get_pdf_page_count, iter_pdf_pages
from app.utils import ensure_dir, get_translation_mode_config, get_translation_mode_labels


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
require_auth()
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

with st.sidebar:
    st.markdown("### Performance")
    perf_mode_label = st.selectbox(
        "Modo de processamento",
        ["Rapido", "Equilibrado", "Qualidade"],
        index=1,
    )
    perf_mode_map = {"Rapido": "fast", "Equilibrado": "balanced", "Qualidade": "quality"}
    performance_mode = perf_mode_map[perf_mode_label]

    use_translation_cache = st.checkbox("Usar cache de traducao", value=True)
    debug_enabled = not st.checkbox("Desativar debug para acelerar", value=True)

    st.markdown("### Configuracao de Fonte")
    min_font_size = st.slider("Tamanho minimo da fonte", min_value=6, max_value=24, value=9, step=1)
    max_font_size = st.slider("Tamanho maximo da fonte", min_value=12, max_value=64, value=32, step=1)
    if min_font_size > max_font_size:
        min_font_size = max_font_size
    line_spacing_ratio = st.slider("Espacamento entre linhas", min_value=0.8, max_value=2.0, value=1.12, step=0.02)
    auto_font_resize = st.checkbox("Ajuste automatico da fonte", value=True)
    center_text = st.checkbox("Centralizar texto", value=True)
    bold_text = st.checkbox("Negrito", value=False)
    text_color_hex = st.color_picker("Cor da fonte", value="#000000")
    text_color = (
        int(text_color_hex[1:3], 16),
        int(text_color_hex[3:5], 16),
        int(text_color_hex[5:7], 16),
    )

translation_style = "natural"
translation_style_label = "Natural"

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

all_batch_files: list[tuple[str, bytes]] = []
page_display_info: list[tuple[int, str]] = []

global_page_counter = 0
for file_idx, file_item in enumerate(inputs, start=1):
    file_kind = file_item["kind"]
    file_name = file_item["name"]
    file_content = file_item["content"]

    if file_kind == "pdf_error":
        global_page_counter += 1
        page_display_info.append((global_page_counter, file_name))
        continue

    if file_kind == "image":
        global_page_counter += 1
        all_batch_files.append((file_name, file_content))
        page_display_info.append((global_page_counter, file_name))
        continue

    temp_pdf_dir = ensure_dir(output_pdf_pages_dir / f"pdf_{file_idx}_{file_item['hash']}")
    for page_number_in_file, total_pdf_pages, page_path in iter_pdf_pages(file_content, temp_pdf_dir, dpi=220):
        global_page_counter += 1
        page_bytes = b""
        try:
            page_bytes = page_path.read_bytes()
        except Exception:
            pass
        if page_bytes:
            page_name = f"{Path(file_name).stem}_page_{page_number_in_file:04d}.png"
            all_batch_files.append((page_name, page_bytes))
            page_display_info.append((global_page_counter, page_name))
        if page_path.exists():
            page_path.unlink(missing_ok=True)

if not all_batch_files:
    status_placeholder.warning("Nenhum arquivo valido para processar.")
    st.stop()

status_placeholder.info(f"Processando {len(all_batch_files)} paginas em lote...")

def update_batch_progress(value: float, message: str, meta=None) -> None:
    overall_progress.progress(int(max(0, min(100, round(value * 100)))))

batch_results = process_batch(
    files=all_batch_files,
    translation_mode=translation_mode,
    source_lang=source_lang,
    target_lang=target_lang,
    ocr_lang=ocr_lang,
    translation_style=translation_style,
    performance_mode=performance_mode,
    debug_enabled=debug_enabled,
    progress_callback=update_batch_progress,
    min_font_size=min_font_size,
    max_font_size=max_font_size,
    line_spacing_ratio=line_spacing_ratio,
    auto_font_resize=auto_font_resize,
    center_text=center_text,
    bold_text=bold_text,
    text_color=text_color,
)

translated_for_zip: list[tuple[int, Path]] = []

for idx, result in enumerate(batch_results):
    page_num = idx + 1
    slot = st.empty()
    translated_path = Path(result.translated_image_path)
    if translated_path.exists():
        render_page_success(slot, page_num, translated_path, f"batch_{page_num}")
        translated_for_zip.append((page_num, translated_path))
    else:
        render_page_error(slot, page_num, "Arquivo traduzido nao encontrado.")

for err_idx in range(len(batch_results), len(all_batch_files)):
    page_num = err_idx + 1
    slot = st.empty()
    render_page_error(slot, page_num, "Pagina nao processada.")

status_placeholder.success(
    f"Finalizado. {len(batch_results)}/{len(all_batch_files)} paginas traduzidas."
)

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
