"""
Manga Translator — Modern UI
Dark theme · Sidebar config · Card-based results · Stage progress
"""

from __future__ import annotations

import os
import zipfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import time
from uuid import uuid4

import streamlit as st

from app.backend import process_uploaded_image
from app.pdf_utils import get_pdf_page_count, iter_pdf_pages
from app.utils import ensure_dir, get_translation_flow_config, get_translation_flow_labels

# ── Constants ──────────────────────────────────────────────────────────
PAGE_TITLE = "Manga Translator"
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "pdf"]
MAX_FILE_SIZE_MB = 200

STAGE_ICONS = {
    "load_image": "📂",
    "detect": "🔍",
    "ocr": "📝",
    "translate": "🌐",
    "clean": "🧹",
    "render": "✨",
    "finish": "✅",
}

STAGE_LABELS_PT = {
    "load_image": "Carregando",
    "detect": "Detectando balões",
    "ocr": "Lendo texto",
    "translate": "Traduzindo",
    "clean": "Limpando",
    "render": "Renderizando",
    "finish": "Finalizado",
}


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def file_hash(content: bytes) -> str:
    digest = sha256()
    digest.update(content)
    return digest.hexdigest()


def page_job_key(
    file_hash_value, file_name, page_number, translation_flow,
    source_lang, target_lang, ocr_lang, ocr_engine,
    performance_mode, translation_style, translation_provider,
) -> str:
    digest = sha256()
    digest.update(str(file_hash_value).encode("utf-8", errors="ignore"))
    digest.update(clean_text(file_name).encode("utf-8", errors="ignore"))
    digest.update(f"_page_{page_number}_".encode("utf-8"))
    for v in [translation_flow, source_lang, target_lang, ocr_lang,
              ocr_engine, performance_mode, translation_style, translation_provider]:
        digest.update(clean_text(v).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def init_state() -> None:
    defaults = {
        "translated_pages": {},
        "translation_errors": {},
        "jobs": {},
        "active_job_id": None,
        "app_runs": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_job(total_pages: int) -> str:
    job_id = uuid4().hex[:12]
    st.session_state["jobs"][job_id] = {
        "id": job_id,
        "created_at": time(),
        "total_pages": total_pages,
        "current_page": 0,
        "current_stage": "pending",
        "status": "pending",
        "pages": {
            idx: {"page": idx, "status": "pending", "stage": "", "error": ""}
            for idx in range(1, total_pages + 1)
        },
    }
    return job_id


def set_page_stage(job_id: str, page: int, stage: str, status: str = "") -> None:
    job = st.session_state.get("jobs", {}).get(job_id)
    if not job:
        return
    p = job["pages"].setdefault(page, {"page": page, "status": "pending", "stage": "", "error": ""})
    p["stage"] = clean_text(stage) or p["stage"]
    if status:
        p["status"] = status
    job["current_page"] = max(job.get("current_page", 0), page)
    job["current_stage"] = STAGE_LABELS_PT.get(stage, stage)


def set_job_status(job_id: str, status: str) -> None:
    job = st.session_state.get("jobs", {}).get(job_id)
    if job:
        job["status"] = status


def normalize_cache_entry(value):
    if value is None:
        return None
    if isinstance(value, dict):
        tp = clean_text(value.get("translated_path"))
        if tp:
            return {"translated_path": tp, "timings": value.get("timings") or {}}
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
    return " · ".join(parts)


def validate_pipeline_result(result) -> None:
    metadata = getattr(result, "metadata", {}) or {}
    translated_count = int(metadata.get("translated_count") or 0)
    rendered_count = int(metadata.get("rendered_count") or 0)
    detected_count = int(metadata.get("detected_count") or metadata.get("bubble_count") or 0)
    if translated_count > 0 and rendered_count <= 0:
        skipped = metadata.get("skipped_bubbles") or []
        details = "; ".join(clean_text(item.get("notes")) for item in skipped if isinstance(item, dict))
        message = "Texto traduzido mas limpeza/renderização falhou em todos os balões."
        if details:
            message = f"{message} Motivos: {details}"
        raise RuntimeError(message)
    elif detected_count > 0 and translated_count <= 0:
        raise RuntimeError("Balões detectados, mas nenhuma tradução válida foi gerada para renderizar.")


def build_zip_bytes(items: list[tuple[int, Path]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for page_number, translated_path in items:
            arcname = f"page_{int(page_number):04d}.png"
            archive.writestr(arcname, translated_path.read_bytes())
    buffer.seek(0)
    return buffer.getvalue()


def gather_inputs(uploaded_files) -> tuple[list[dict], int]:
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
                files_data.append({
                    "kind": "pdf_error", "name": raw_name, "content": content,
                    "hash": hash_value, "page_count": count, "error": clean_text(exc) or "Falha ao abrir PDF.",
                })
                total_pages += count
                continue
            files_data.append({
                "kind": "pdf", "name": raw_name, "content": content,
                "hash": hash_value, "page_count": max(0, int(count)), "error": "",
            })
            total_pages += max(0, int(count))
            continue

        files_data.append({
            "kind": "image", "name": raw_name, "content": content,
            "hash": hash_value, "page_count": 1, "error": "",
        })
        total_pages += 1

    return files_data, total_pages


# ── Dark Theme CSS ─────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Global ── */
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
.stApp {background: #0e1117; color: #e1e8f0;}

/* ── Sidebar ── */
[data-testid="stSidebar"] {background: #161b22; border-right: 1px solid #30363d;}

/* ── Upload zone ── */
[data-testid="stFileUploader"] > div {
    background: #1c2333 !important;
    border: 2px dashed #388bfd !important;
    border-radius: 12px !important;
    padding: 2rem !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #58a6ff !important;
    background: #1f2a3a !important;
}

/* ── Result cards ── */
.page-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.page-card:hover {border-color: #388bfd;}

/* ── Stage progress badges ── */
.stage-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.stage-pending {background: #21262d; color: #8b949e;}
.stage-active {background: #1f3a5f; color: #58a6ff; animation: pulse 1.5s infinite;}
.stage-done {background: #1a3a2a; color: #3fb950;}

@keyframes pulse {
    0%, 100% {opacity: 1;}
    50% {opacity: 0.7;}
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #21262d;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    background: #1f2a3a !important;
    color: #58a6ff !important;
}

/* ── Buttons ── */
.stDownloadButton > button {
    background: #238636 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.big-zip-btn .stDownloadButton > button {
    background: #21262d !important;
    border: 2px solid #388bfd !important;
    color: #58a6ff !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    height: auto !important;
}

/* ── Select boxes ── */
[data-baseweb="select"] > div {
    background: #21262d !important;
    border-color: #30363d !important;
    border-radius: 8px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: #388bfd !important;
    border-radius: 4px !important;
}

/* ── Timing info ── */
.timing-badge {
    font-size: 0.72rem;
    color: #8b949e;
    font-family: monospace;
}

/* ── Info/warning boxes ── */
.stAlert {border-radius: 8px !important;}

/* ── Images ── */
[data-testid="stImage"] img {border-radius: 8px;}
</style>
"""


# ── Render Functions ───────────────────────────────────────────────────

def render_job_status(job_id: str) -> None:
    """Render a compact job status bar with stage badges."""
    job = st.session_state.get("jobs", {}).get(job_id)
    if not job:
        return

    total = job.get("total_pages", 0)
    current = job.get("current_page", 0)
    current_stage = job.get("current_stage", "pending")

    # Progress bar
    if total > 0:
        pct = current / total
        st.progress(pct, text=f"📖 Página {current}/{total} · {current_stage}")

    # Stage badges
    stages_order = ["load_image", "detect", "ocr", "translate", "clean", "render", "finish"]
    badge_html = ""
    for s in stages_order:
        icon = STAGE_ICONS.get(s, "⏳")
        label = STAGE_LABELS_PT.get(s, s)
        if current_stage == s:
            cls = "stage-active"
        elif stages_order.index(s) < stages_order.index(current_stage) if current_stage in stages_order else False:
            cls = "stage-done"
        else:
            cls = "stage-pending"
        badge_html += f'<span class="stage-badge {cls}">{icon} {label}</span> '

    st.markdown(badge_html, unsafe_allow_html=True)


def render_result_card(slot, page_number: int, translated_path: Path, job_key: str, timing_str: str = "") -> None:
    with slot.container():
        cols = st.columns([1, 3])
        with cols[0]:
            st.markdown(f"**📄 Página {page_number}**")
            if timing_str:
                st.markdown(f'<span class="timing-badge">⏱ {timing_str}</span>', unsafe_allow_html=True)
        with cols[1]:
            st.image(str(translated_path), use_container_width=True)
        with cols[0]:
            with open(translated_path, "rb") as f:
                st.download_button(
                    "⬇️ Baixar",
                    data=f,
                    file_name=translated_path.name,
                    mime="image/png",
                    key=f"dl_{job_key}",
                    use_container_width=True,
                )
        st.divider()


def render_error_card(slot, page_number: int, message: str) -> None:
    with slot.container():
        st.markdown(f"**📄 Página {page_number}**")
        st.error(f"Não foi possível traduzir a página {page_number}.")
        if clean_text(message):
            with st.expander("🔧 Detalhes técnicos"):
                st.code(clean_text(message), language="text")
        st.divider()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0">', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        /* Mobile tweaks */
        @media (max-width: 600px) {
            .stButton>button { min-height: 44px; font-size: 1.1rem; }
            .stDownloadButton>button { min-height: 44px; font-size: 1.1rem; }
            .stSelectbox>div>div { min-height: 44px; }
            .stFileUploader>div { min-height: 100px; }
            .stImage img { border-radius: 4px; }
            .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    init_state()

    # ── Header ──
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.markdown("<h1 style='font-size:2.5rem; margin-top:0.5rem;'>📚</h1>", unsafe_allow_html=True)
    with col_title:
        st.markdown(
            '<h1 style="font-size:1.8rem; margin-bottom:0; background: linear-gradient(90deg, #58a6ff, #a371f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Manga Translator</h1>',
            unsafe_allow_html=True,
        )
        st.caption("Tradução automática de mangá · Upload de imagens ou PDF · Página por página")

    st.divider()

    # ── Sidebar config ──
    mode_labels = get_translation_flow_labels()

    performance_labels = {"Rápido": "fast", "Equilibrado": "balanced", "Qualidade": "quality"}

    with st.sidebar:
        sel_mode = st.selectbox(
            "🌐 Modo de tradução",
            list(mode_labels.keys()),
            index=0,
            help="Idioma de origem → destino",
        )
        # OCR options por modo: manga só faz sentido pra japonês
        sel_mode_config = get_translation_flow_config(sel_mode)
        sel_source_lang = sel_mode_config["source_lang"]
        is_cjk = sel_source_lang in ("auto", "ja", "zh", "ko")
        if is_cjk:
            ocr_options = {"Manga OCR": "manga", "PaddleOCR": "paddle"}
            default_ocr = "Manga OCR"
        else:
            ocr_options = {"PaddleOCR": "paddle", "Automático": "auto"}
            default_ocr = "PaddleOCR"
        default_idx = list(ocr_options.keys()).index(default_ocr) if default_ocr in ocr_options else 0
        sel_ocr_label = st.selectbox(
            "📝 Motor de OCR",
            list(ocr_options.keys()),
            index=default_idx,
            help="Manga OCR é recomendado para texto japonês de mangá",
        )

        default_perf_idx = list(performance_labels.keys()).index("Rápido")
        sel_perf_label = st.selectbox(
            "⚡ Desempenho",
            list(performance_labels.keys()),
            index=default_perf_idx,
            help="Equilíbrio entre velocidade e qualidade",
        )

        st.divider()
        st.markdown("##### ℹ️ Sobre")
        st.caption(
            "Motor de tradução: usa tradução leve por padrão para não sobrecarregar VPS pequena.\n\n"
            "Configure via env: `TRANSLATION_PROVIDER`, `TRANSLATION_MODEL`, "
            "`DEFAULT_SOURCE_LANG`, `DEFAULT_TARGET_LANG`."
        )

    selected_mode = sel_mode
    selected_mode_config = get_translation_flow_config(selected_mode)
    source_lang = selected_mode_config["source_lang"]
    target_lang = selected_mode_config["target_lang"]
    ocr_lang = selected_mode_config["ocr_lang"]
    translation_flow = selected_mode_config["mode"]
    translation_flow_label = selected_mode_config["label"]
    ocr_engine = ocr_options[sel_ocr_label]
    performance_mode = performance_labels[sel_perf_label]
    translation_style = "natural"
    debug_enabled = False
    translation_provider = os.getenv("TRANSLATION_PROVIDER", "multilingual")

    # ── Two-column layout: upload | results ──
    col_upload, col_results = st.columns([2, 3], gap="large")

    with col_upload:
        st.subheader("📤 Upload")
        uploaded_files = st.file_uploader(
            "Arraste arquivos ou clique para selecionar",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            total_mb = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
            st.caption(f"📦 {len(uploaded_files)} arquivo(s) · {total_mb:.1f} MB total")

    if not uploaded_files:
        with col_results:
            st.info("👈 Envie imagens ou PDF para iniciar a tradução.")

            # Show stats if we have previous runs
            total_cached = len(st.session_state.get("translated_pages", {}))
            if total_cached > 0:
                st.divider()
                st.markdown("### 📊 Sessão")
                st.caption(f"{total_cached} página(s) em cache nesta sessão.")
        st.stop()

    # ── Start translation ──
    with col_results:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

    inputs, total_pages = gather_inputs(uploaded_files)
    if total_pages <= 0:
        col_results.warning("Nenhuma página válida encontrada.")
        st.stop()

    job_id = create_job(total_pages)
    if st.session_state.get("active_job_id"):
        col_results.warning("⏳ Processamento em andamento. Aguarde terminar.")
        st.stop()

    st.session_state["active_job_id"] = job_id
    set_job_status(job_id, "processing")

    output_pdf_pages_dir = ensure_dir(Path("output") / "_pdf_pages")
    try:
        _pdf_dpi_fast = int(os.getenv("PDF_DPI_FAST", "110") or "110")
        _pdf_dpi_default = int(os.getenv("PDF_DPI_DEFAULT", "130") or "130")
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
            set_page_stage(job_id, global_page_counter, "finish", "failed")
            slot = col_results.empty()
            jk = page_job_key(hash_value, file_name, 1, translation_flow, source_lang, target_lang,
                              ocr_lang, ocr_engine, performance_mode, translation_style, translation_provider)
            st.session_state["translation_errors"][jk] = {"message": file_item["error"]}
            status_placeholder.warning(f"Não foi possível abrir o PDF {file_name}.")
            render_error_card(slot, global_page_counter, file_item["error"])
            progress_bar.progress(int((finished_pages / total_pages) * 100))
            continue

        # ── Process pages in file ──
        def make_iter_page():
            if file_kind == "pdf":
                yield from iter_pdf_pages(file_content, dpi=pdf_dpi)
            else:
                yield file_content

        for page_content in make_iter_page():
            global_page_counter += 1
            set_page_stage(job_id, global_page_counter, "load_image", "processing")
            render_job_status(job_id)

            jk = page_job_key(
                hash_value, file_name, 1, translation_flow,
                source_lang, target_lang, ocr_lang, ocr_engine,
                performance_mode, translation_style, translation_provider,
            )
            slot = col_results.empty()

            # Check cache
            cached_result = normalize_cache_entry(st.session_state["translated_pages"].get(jk))
            cached_error = st.session_state["translation_errors"].get(jk)

            if cached_result is not None:
                translated_path = Path(cached_result["translated_path"])
                if translated_path.exists():
                    set_page_stage(job_id, global_page_counter, "finish", "done")
                    render_result_card(slot, global_page_counter, translated_path, jk,
                                       format_timings(cached_result.get("timings")))
                    translated_for_zip.append((global_page_counter, translated_path))
                    finished_pages += 1
                    progress_bar.progress(int((finished_pages / total_pages) * 100))
                    continue
                else:
                    st.session_state["translated_pages"].pop(jk, None)
                    st.session_state["translation_errors"][jk] = {"message": "Arquivo traduzido não encontrado no cache."}

            if cached_error is not None:
                set_page_stage(job_id, global_page_counter, "finish", "failed")
                render_error_card(slot, global_page_counter, clean_text(cached_error.get("message")))
                finished_pages += 1
                progress_bar.progress(int((finished_pages / total_pages) * 100))
                continue

            status_placeholder.info(
                f"📖 Página {global_page_counter}/{total_pages} · {translation_flow_label}"
            )

            def update_progress(value: float, message: str, meta=None) -> None:
                stage_id = clean_text((meta or {}).get("stage_id")) if isinstance(meta, dict) else ""
                stage_map = {
                    "load_image": "load_image", "detect": "detect", "ocr": "ocr",
                    "translate": "translate", "clean": "clean", "render": "render",
                    "finish": "finish",
                }
                mapped = stage_map.get(stage_id, "")
                if mapped:
                    set_page_stage(job_id, global_page_counter, mapped)
                local_value = float(max(0.0, min(1.0, value)))
                absolute_progress = ((finished_pages + local_value) / total_pages) * 100.0
                progress_bar.progress(int(max(0, min(100, round(absolute_progress)))))

            try:
                result = process_uploaded_image(
                    filename=file_name, content=file_content,
                    translation_flow=translation_flow, source_lang=source_lang,
                    target_lang=target_lang, ocr_lang=ocr_lang, ocr_engine=ocr_engine,
                    translation_style=translation_style, performance_mode=performance_mode,
                    translation_provider=translation_provider, debug_enabled=debug_enabled,
                    progress_callback=update_progress,
                )
                validate_pipeline_result(result)
                set_page_stage(job_id, global_page_counter, "finish", "done")

                translated_path = Path(result.translated_image_path)
                timings = result.metadata.get("timings", {}) if isinstance(result.metadata, dict) else {}
                st.session_state["translated_pages"][jk] = {
                    "translated_path": str(translated_path),
                    "timings": timings,
                }

                render_result_card(slot, global_page_counter, translated_path, jk, format_timings(timings))
                translated_for_zip.append((global_page_counter, translated_path))

            except RuntimeError as exc:
                set_page_stage(job_id, global_page_counter, "finish", "failed")
                st.session_state["translation_errors"][jk] = {"message": str(exc)}
                render_error_card(slot, global_page_counter, str(exc))

            except Exception as exc:
                set_page_stage(job_id, global_page_counter, "finish", "failed")
                st.session_state["translation_errors"][jk] = {"message": str(exc)}
                render_error_card(slot, global_page_counter, str(exc))

            finished_pages += 1
            progress_bar.progress(int((finished_pages / total_pages) * 100))

    # ── All done ──
    set_job_status(job_id, "done")
    st.session_state["active_job_id"] = None

    st.divider()

    # ── Download all ──
    if translated_for_zip:
        zip_bytes = build_zip_bytes(translated_for_zip)
        dl_col = st.container()
        dl_col.markdown('<div class="big-zip-btn">', unsafe_allow_html=True)
        dl_col.download_button(
            "📦 Baixar todas as páginas (ZIP)",
            data=zip_bytes,
            file_name="manga_translated.zip",
            mime="application/zip",
            use_container_width=True,
        )
        dl_col.markdown("</div>", unsafe_allow_html=True)

        # Stats
        ok_count = len(translated_for_zip)
        err_count = total_pages - ok_count
        cols_stats = st.columns(3)
        cols_stats[0].metric("✅ Traduzidas", ok_count)
        cols_stats[1].metric("❌ Erros", err_count)
        cols_stats[2].metric("📄 Total", total_pages)


if __name__ == "__main__":
    main()
