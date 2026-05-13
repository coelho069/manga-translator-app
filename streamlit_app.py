from __future__ import annotations

import traceback
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from app.backend import process_uploaded_image


EXAMPLES_DIR = Path("examples") / "bubbles"
LANG_OPTIONS = {"Inglês": "en", "Chinês": "zh-CN"}
STYLE_OPTIONS = {"Natural": "natural", "Literal": "literal"}
PERFORMANCE_OPTIONS = {"Equilibrado": "balanced", "Rápido": "fast", "Qualidade": "quality"}

PROCESS_STEPS = [
    {
        "id": "load_image",
        "emoji": "🖼️",
        "title": "Imagem",
        "short": "Carregando a página",
        "timing_key": "load_image",
    },
    {
        "id": "detect",
        "emoji": "💬",
        "title": "Balões",
        "short": "Encontrando falas",
        "timing_key": "detect",
    },
    {
        "id": "ocr",
        "emoji": "🔎",
        "title": "OCR",
        "short": "Lendo textos",
        "timing_key": "ocr",
    },
    {
        "id": "translate",
        "emoji": "🌐",
        "title": "Tradução",
        "short": "Convertendo para PT-BR",
        "timing_key": "translate",
    },
    {
        "id": "clean",
        "emoji": "🧽",
        "title": "Limpeza",
        "short": "Apagando o original",
        "timing_key": "clean",
    },
    {
        "id": "render",
        "emoji": "✍️",
        "title": "Edição",
        "short": "Inserindo tradução",
        "timing_key": "render",
    },
    {
        "id": "finish",
        "emoji": "✅",
        "title": "Final",
        "short": "Preparando resultado",
        "timing_key": "save",
    },
]
STAGE_ORDER = [step["id"] for step in PROCESS_STEPS]
MESSAGE_STAGE_HINTS = {
    "carregando": "load_image",
    "detectando": "detect",
    "baloes": "detect",
    "balões": "detect",
    "lendo": "ocr",
    "ocr": "ocr",
    "traduzindo": "translate",
    "apagando": "clean",
    "limpando": "clean",
    "inserindo": "render",
    "renderizando": "render",
    "finalizando": "finish",
    "finalizado": "finish",
    "nenhum balao": "finish",
    "nenhum balão": "finish",
}
STATUS_LABELS = {
    "pending": "Pendente",
    "running": "Em andamento",
    "done": "Concluído",
    "error": "Erro",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #090d14;
            --panel: rgba(18, 24, 36, 0.90);
            --panel-strong: rgba(25, 33, 49, 0.96);
            --panel-soft: rgba(255, 255, 255, 0.055);
            --stroke: rgba(255, 255, 255, 0.12);
            --stroke-strong: rgba(255, 255, 255, 0.21);
            --text: #f4f7fb;
            --muted: #a7b3c7;
            --accent: #8b7cff;
            --accent-2: #2ee6a6;
            --accent-3: #ffd166;
            --danger: #ff6680;
            --warning: #f3b84b;
            --shadow: 0 18px 55px rgba(0, 0, 0, 0.36);
            --radius: 18px;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 8%, rgba(139, 124, 255, 0.18), transparent 30%),
                radial-gradient(circle at 88% 12%, rgba(46, 230, 166, 0.12), transparent 28%),
                linear-gradient(180deg, #090d14 0%, #0d1320 54%, #090d14 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, p, label, span {
            letter-spacing: 0;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--stroke) !important;
            border-radius: var(--radius) !important;
            background: var(--panel) !important;
            box-shadow: var(--shadow);
            transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            border-color: var(--stroke-strong) !important;
            background: var(--panel-strong) !important;
        }

        .app-hero {
            border: 1px solid var(--stroke);
            border-radius: 26px;
            padding: 30px 32px;
            background:
                linear-gradient(135deg, rgba(139, 124, 255, 0.22), rgba(46, 230, 166, 0.09)),
                rgba(18, 25, 40, 0.80);
            box-shadow: var(--shadow);
            animation: fadeUp 360ms ease-out both;
            overflow: hidden;
            position: relative;
        }

        .app-hero:after {
            content: "";
            position: absolute;
            inset: auto -12% -35% auto;
            width: 360px;
            height: 360px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 209, 102, 0.12), transparent 64%);
            pointer-events: none;
        }

        .eyebrow {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .app-hero h1 {
            color: var(--text);
            font-size: clamp(2.1rem, 4vw, 3.5rem);
            line-height: 1.05;
            margin: 0 0 0.8rem 0;
        }

        .app-hero p {
            color: var(--muted);
            max-width: 820px;
            font-size: 1.04rem;
            line-height: 1.65;
            margin: 0;
        }

        .hero-badges,
        .metric-grid,
        .timeline-grid {
            display: grid;
            gap: 12px;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }

        .badge,
        .mini-badge {
            border: 1px solid var(--stroke);
            border-radius: 999px;
            color: #e2e9f8;
            background: rgba(255, 255, 255, 0.07);
        }

        .badge {
            padding: 7px 12px;
            font-size: 0.84rem;
        }

        .mini-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 9px;
            font-size: 0.76rem;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .section-title {
            margin: 2rem 0 0.8rem;
        }

        .section-title h2 {
            color: var(--text);
            font-size: 1.3rem;
            margin: 0 0 0.2rem;
        }

        .section-title p {
            color: var(--muted);
            margin: 0;
            line-height: 1.55;
        }

        .empty-state {
            border: 1px dashed rgba(255, 255, 255, 0.22);
            border-radius: var(--radius);
            background: rgba(255, 255, 255, 0.045);
            padding: 28px;
            color: var(--muted);
            text-align: center;
            animation: pulseBorder 2.8s ease-in-out infinite;
        }

        .empty-state strong {
            display: block;
            color: var(--text);
            font-size: 1.04rem;
            margin-bottom: 6px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(46, 230, 166, 0.28);
            color: #c8fff1;
            background: rgba(46, 230, 166, 0.10);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.9rem;
            margin-bottom: 12px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--accent-2);
            box-shadow: 0 0 18px rgba(46, 230, 166, 0.75);
            animation: breathe 1.3s ease-in-out infinite;
        }

        .progress-shell {
            border: 1px solid var(--stroke);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(139, 124, 255, 0.12), rgba(46, 230, 166, 0.06)),
                rgba(255, 255, 255, 0.045);
            padding: 18px;
            margin-bottom: 14px;
            animation: fadeUp 260ms ease-out both;
        }

        .progress-kicker {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .progress-main {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 16px;
            margin-top: 6px;
        }

        .progress-main h3 {
            margin: 0;
            color: var(--text);
            font-size: 1.28rem;
        }

        .progress-main p {
            color: var(--muted);
            margin: 4px 0 0;
        }

        .progress-percent {
            min-width: 86px;
            text-align: right;
            color: var(--accent-3);
            font-size: 2rem;
            font-weight: 900;
        }

        .timeline-grid {
            grid-template-columns: repeat(7, minmax(0, 1fr));
            margin: 14px 0 4px;
        }

        .stage-card {
            border: 1px solid var(--stroke);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.045);
            padding: 12px;
            min-height: 136px;
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }

        .stage-card.done {
            border-color: rgba(46, 230, 166, 0.36);
            background: rgba(46, 230, 166, 0.08);
        }

        .stage-card.running {
            border-color: rgba(255, 209, 102, 0.54);
            background: rgba(255, 209, 102, 0.095);
            box-shadow: 0 0 28px rgba(255, 209, 102, 0.08);
            animation: cardPulse 1.45s ease-in-out infinite;
        }

        .stage-card.error {
            border-color: rgba(255, 102, 128, 0.62);
            background: rgba(255, 102, 128, 0.09);
        }

        .stage-emoji {
            font-size: 1.45rem;
            line-height: 1;
        }

        .stage-title {
            color: var(--text);
            font-weight: 850;
            margin-top: 8px;
        }

        .stage-short,
        .stage-time {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .stage-status {
            display: inline-flex;
            margin-top: 10px;
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 0.70rem;
            font-weight: 800;
            color: rgba(255, 255, 255, 0.88);
            background: rgba(255, 255, 255, 0.09);
        }

        .stage-card.done .stage-status {
            color: #d8fff3;
            background: rgba(46, 230, 166, 0.18);
        }

        .stage-card.running .stage-status {
            color: #fff2c8;
            background: rgba(255, 209, 102, 0.18);
        }

        .stage-card.error .stage-status {
            color: #ffdbe2;
            background: rgba(255, 102, 128, 0.18);
        }

        .metric-grid {
            grid-template-columns: repeat(5, minmax(0, 1fr));
            margin: 12px 0 10px;
        }

        .metric-card {
            border: 1px solid var(--stroke);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.045);
            padding: 14px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.45rem;
            font-weight: 900;
            margin-top: 4px;
        }

        .result-card-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }

        .result-card-title h4 {
            margin: 0;
            color: var(--text);
        }

        .bubble-row {
            border: 1px solid var(--stroke);
            border-radius: 16px;
            padding: 14px 15px;
            background: rgba(255, 255, 255, 0.045);
            margin-bottom: 12px;
            transition: transform 160ms ease, border-color 160ms ease;
        }

        .bubble-row:hover {
            transform: translateY(-1px);
            border-color: rgba(139, 124, 255, 0.42);
        }

        .bubble-title {
            color: var(--text);
            font-weight: 850;
            margin-bottom: 8px;
        }

        .bubble-label {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-top: 8px;
        }

        .bubble-text {
            color: #dce6f8;
            line-height: 1.5;
            margin: 2px 0 8px;
        }

        .friendly-error {
            border: 1px solid rgba(255, 102, 128, 0.40);
            border-radius: 18px;
            background: rgba(255, 102, 128, 0.08);
            padding: 18px;
            margin-bottom: 12px;
        }

        .friendly-error h3 {
            margin: 0 0 8px;
            color: #ffdbe2;
        }

        .friendly-error p,
        .friendly-error li {
            color: #f2bdc8;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border: 0 !important;
            border-radius: 14px !important;
            color: #ffffff !important;
            background: linear-gradient(135deg, #8b7cff, #2ee6a6) !important;
            box-shadow: 0 12px 30px rgba(139, 124, 255, 0.25);
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
            font-weight: 850 !important;
            min-height: 46px;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-2px);
            filter: brightness(1.06);
            box-shadow: 0 16px 38px rgba(46, 230, 166, 0.20);
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(139, 124, 255, 0.42);
            border-radius: var(--radius);
            background: rgba(139, 124, 255, 0.06);
            padding: 10px;
            transition: border-color 160ms ease, background 160ms ease;
        }

        div[data-testid="stFileUploader"]:hover {
            border-color: rgba(46, 230, 166, 0.58);
            background: rgba(46, 230, 166, 0.06);
        }

        div[data-testid="stProgress"] > div > div > div {
            background:
                linear-gradient(90deg, #8b7cff, #2ee6a6, #ffd166, #2ee6a6) !important;
            background-size: 220% 100% !important;
            animation: progressShimmer 1.7s linear infinite;
            transition: width 220ms ease !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid var(--stroke);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes breathe {
            0%, 100% { opacity: 0.55; transform: scale(0.85); }
            50% { opacity: 1; transform: scale(1.1); }
        }

        @keyframes pulseBorder {
            0%, 100% { border-color: rgba(255, 255, 255, 0.16); }
            50% { border-color: rgba(139, 124, 255, 0.45); }
        }

        @keyframes progressShimmer {
            from { background-position: 0% 50%; }
            to { background-position: 220% 50%; }
        }

        @keyframes cardPulse {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-3px); }
        }

        @media (max-width: 980px) {
            .timeline-grid,
            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .app-hero {
                padding: 24px 20px;
            }
            .timeline-grid,
            .metric-grid {
                grid-template-columns: 1fr;
            }
            .progress-main {
                align-items: flex-start;
                flex-direction: column;
            }
            .progress-percent {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Manga translation quest</div>
            <h1>Manga Translator App</h1>
            <p>
                Envie uma página de mangá, acompanhe cada etapa da tradução e receba a versão final
                em português com os balões limpos e editados.
            </p>
            <div class="hero-badges">
                <span class="badge">💬 YOLO segmentation</span>
                <span class="badge">🔎 PaddleOCR</span>
                <span class="badge">🌐 PT-BR natural</span>
                <span class="badge">🖥️ Execução local</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{escape(title)}</h2>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill_html(text: str) -> str:
    return f"""
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>{escape(clean_text(text))}</span>
        </div>
        """


def status_pill(text: str) -> None:
    st.markdown(status_pill_html(text), unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{escape(title)}</strong>
            <span>{escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_stage_state(default_status: str = "pending") -> dict[str, str]:
    return {stage_id: default_status for stage_id in STAGE_ORDER}


def infer_stage_id(message: str, meta: dict[str, Any] | None = None) -> str:
    if meta:
        stage_id = clean_text(meta.get("stage_id"))
        if stage_id in STAGE_ORDER:
            return stage_id
    lowered = clean_text(message).lower()
    for hint, stage_id in MESSAGE_STAGE_HINTS.items():
        if hint in lowered:
            return stage_id
    return "load_image"


def update_stage_state(stage_state: dict[str, str], stage_id: str, status: str = "running") -> None:
    if stage_id not in STAGE_ORDER:
        return
    active_index = STAGE_ORDER.index(stage_id)
    for step_id in STAGE_ORDER[:active_index]:
        if stage_state.get(step_id) != "error":
            stage_state[step_id] = "done"
    if status == "done":
        for step_id in STAGE_ORDER[: active_index + 1]:
            if stage_state.get(step_id) != "error":
                stage_state[step_id] = "done"
    else:
        stage_state[stage_id] = status


def mark_all_done(stage_state: dict[str, str]) -> None:
    for step_id in STAGE_ORDER:
        stage_state[step_id] = "done"


def mark_stage_error(stage_state: dict[str, str], stage_id: str) -> None:
    if stage_id not in STAGE_ORDER:
        stage_id = "finish"
    update_stage_state(stage_state, stage_id, "error")
    stage_state[stage_id] = "error"


def format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f}s"


def render_timeline(stage_state: dict[str, str], active_stage_id: str | None = None, timings: dict[str, Any] | None = None) -> str:
    timings = timings or {}
    cards = []
    for step in PROCESS_STEPS:
        stage_id = step["id"]
        status = stage_state.get(stage_id, "pending")
        status_label = STATUS_LABELS.get(status, "Pendente")
        timing = format_seconds(timings.get(step["timing_key"]))
        if stage_id == "finish" and not timing:
            timing = format_seconds(timings.get("total"))
        active_marker = " • agora" if active_stage_id == stage_id and status == "running" else ""
        cards.append(
            f"""
            <div class="stage-card {escape(status)}">
                <div class="stage-emoji">{step["emoji"]}</div>
                <div class="stage-title">{escape(step["title"])}</div>
                <div class="stage-short">{escape(step["short"])}</div>
                <div class="stage-time">{escape(timing or "aguardando")}</div>
                <span class="stage-status">{escape(status_label + active_marker)}</span>
            </div>
            """
        )
    return f'<div class="timeline-grid">{"".join(cards)}</div>'


def render_progress_header(percent: int, stage_label: str, message: str) -> str:
    return f"""
    <div class="progress-shell">
        <div class="progress-kicker">Progresso do processamento</div>
        <div class="progress-main">
            <div>
                <h3>{escape(stage_label)}</h3>
                <p>{escape(message)}</p>
            </div>
            <div class="progress-percent">{int(percent)}%</div>
        </div>
    </div>
    """


def render_metric_cards(metrics: list[tuple[str, Any, str]]) -> None:
    cards = []
    for label, value, suffix in metrics:
        display_value = clean_text(value)
        if suffix:
            display_value = f"{display_value}{suffix}"
        cards.append(
            f"""
            <div class="metric-card">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{escape(display_value)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def count_result_items(result) -> dict[str, Any]:
    metadata = result.metadata or {}
    bubbles = result.bubbles or []
    return {
        "detected": metadata.get("detected_count", metadata.get("bubble_count", len(bubbles))),
        "ocr": metadata.get("ocr_count", sum(1 for bubble in bubbles if bubble.source_text)),
        "translated": metadata.get("translated_count", sum(1 for bubble in bubbles if bubble.translated_text)),
        "rendered": metadata.get("rendered_count", sum(1 for bubble in bubbles if bubble.translated_text)),
        "cleaned": metadata.get("cleanup_success_count", sum(1 for bubble in bubbles if bubble.cleanup_success)),
        "total_time": metadata.get("total_time", (metadata.get("timings") or {}).get("total")),
    }


def get_input_image(use_example: bool, selected_example, uploaded_file):
    if use_example and selected_example is not None:
        return selected_example.read_bytes(), selected_example.name
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name
    return None, None


def make_job_key(
    content: bytes,
    filename: str,
    source_lang: str,
    translation_style: str,
    performance_mode: str,
    debug_enabled: bool,
) -> str:
    digest = sha256()
    digest.update(content)
    digest.update(str(filename).encode("utf-8", errors="ignore"))
    digest.update(source_lang.encode("utf-8"))
    digest.update(translation_style.encode("utf-8"))
    digest.update(performance_mode.encode("utf-8"))
    digest.update(str(bool(debug_enabled)).encode("utf-8"))
    return digest.hexdigest()


def bubble_info_card(bubble, source_label: str) -> None:
    source_text = escape(clean_text(bubble.source_text) or "(sem texto detectado)")
    translated_text = escape(clean_text(bubble.translated_text) or "(sem tradução)")
    source_label = escape(source_label)
    notes = getattr(bubble, "processing_notes", []) or []
    notes_html = ""
    if notes:
        joined_notes = escape("; ".join(clean_text(note) for note in notes if clean_text(note)))
        notes_html = f'<span class="mini-badge">⚠️ {joined_notes}</span>'
    cleanup_badge = "🧽 limpo" if getattr(bubble, "cleanup_success", False) else "🧽 limpeza parcial"
    render_badge = "✍️ traduzido" if clean_text(bubble.translated_text) else "✍️ sem tradução"
    st.markdown(
        f"""
        <div class="bubble-row">
            <div class="bubble-title">💬 Balão {bubble.id}</div>
            <span class="mini-badge">{escape(cleanup_badge)}</span>
            <span class="mini-badge">{escape(render_badge)}</span>
            {notes_html}
            <div class="bubble-label">Original ({source_label})</div>
            <div class="bubble-text">{source_text}</div>
            <div class="bubble-label">Português</div>
            <div class="bubble-text">{translated_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result, source_lang_label: str, performance_label: str) -> None:
    metadata = result.metadata or {}
    timings = metadata.get("timings", {}) or {}
    counts = count_result_items(result)
    total_time = format_seconds(counts["total_time"])
    subtitle = "Compare a página original com a versão editada."
    if total_time:
        subtitle = f"{subtitle} A rodada levou {total_time}."

    section_title("Resultado final", subtitle)
    
    st.markdown(
        f"""
        <span class="mini-badge">🌐 Entrada: {escape(source_lang_label)}</span>
        <span class="mini-badge">🇧🇷 Saída: português</span>
        <span class="mini-badge">⚡ Performance: {escape(performance_label)}</span>
        <span class="mini-badge">🧽 Limpos: {escape(clean_text(counts["cleaned"]))}</span>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="result-card-title"><h4>🖼️ Original</h4><span class="mini-badge">entrada</span></div>', unsafe_allow_html=True)
            st.image(Image.open(result.original_image_path), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="result-card-title"><h4>✅ Traduzida</h4><span class="mini-badge">resultado</span></div>', unsafe_allow_html=True)
            st.image(Image.open(result.translated_image_path), use_container_width=True)

    with open(result.translated_image_path, "rb") as image_file:
        st.download_button(
            "⬇️ Baixar imagem traduzida",
            data=image_file,
            file_name=result.translated_image_path.name,
            mime="image/png",
            use_container_width=True,
        )

    if timings:
        with st.expander("Tempos técnicos por etapa"):
            st.json(timings)

    skipped = metadata.get("skipped_bubbles") or []
    if skipped:
        with st.expander("Avisos do processamento"):
            st.json(skipped)

    section_title("Textos detectados", "Revise o OCR e a tradução aplicada em cada balão.")
    if result.bubbles:
        for bubble in result.bubbles:
            bubble_info_card(bubble, source_lang_label)
    else:
        empty_state(
            "Nenhum balão detectado",
            "Confira se o modelo models/bubble_seg.pt é adequado para esse tipo de página.",
        )


def render_error(exc: Exception, stage_state: dict[str, str], active_stage_id: str, technical_traceback: str) -> None:
    mark_stage_error(stage_state, active_stage_id)
    st.markdown(
        """
        <div class="friendly-error">
            <h3>Não consegui concluir esta tradução.</h3>
            <p>O processamento parou antes do resultado final. Tente uma destas ações rápidas:</p>
            <ul>
                <li>Verifique se o modelo existe em <strong>models/bubble_seg.pt</strong>.</li>
                <li>Confirme se as dependências foram instaladas no ambiente virtual.</li>
                <li>Teste uma imagem com melhor resolução e balões mais nítidos.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(clean_text(exc) or "Não foi possível processar a imagem.")
    with st.expander("Detalhes técnicos"):
        st.code(technical_traceback, language="python")


def show_completion_feedback(job_key: str, from_cache: bool = False) -> None:
    if from_cache:
        if hasattr(st, "toast"):
            st.toast("Resultado reaproveitado da sessão.", icon="♻️")
        return
    celebration_key = f"celebrated_{job_key}"
    if st.session_state.get(celebration_key):
        return
    st.session_state[celebration_key] = True
    if hasattr(st, "toast"):
        st.toast("Tradução concluída. Página pronta para baixar!", icon="✅")
    try:
        st.balloons()
    except Exception:
        pass


st.set_page_config(page_title="Manga Translator App", page_icon="💬", layout="wide")
inject_theme()
render_header()

example_files = sorted(EXAMPLES_DIR.glob("*.png")) if EXAMPLES_DIR.exists() else []

section_title("Entrada e configurações", "Escolha a imagem e ajuste idioma/estilo antes de iniciar a missão.")
settings_col, preview_col = st.columns([0.92, 1.08], gap="large")

with settings_col:
    with st.container(border=True):
        st.markdown("#### 🎮 Configurações da missão")
        source_lang_label = st.selectbox("Idioma de entrada", list(LANG_OPTIONS.keys()))
        source_lang = LANG_OPTIONS[source_lang_label]

        translation_style_label = st.selectbox("Estilo de tradução", list(STYLE_OPTIONS.keys()))
        translation_style = STYLE_OPTIONS[translation_style_label]
        performance_label = st.selectbox("Modo de performance", list(PERFORMANCE_OPTIONS.keys()))
        performance_mode = PERFORMANCE_OPTIONS[performance_label]
        debug_enabled = False

        st.markdown("#### 🖼️ Página do mangá")

        use_example = False
        selected_example = None

        uploaded_file = st.file_uploader(
            "Enviar página de mangá",
            type=["png", "jpg", "jpeg"],
            help="Use PNG, JPG ou JPEG. Para melhores resultados, prefira imagens nítidas e com boa resolução.",
        )

        st.caption("Saída: português. Processamento local em CPU por padrão.")

content, filename = get_input_image(use_example, selected_example, uploaded_file)

with preview_col:
    with st.container(border=True):
        st.markdown("#### Pré-visualização")
        if content is not None and filename is not None:
            status_pill(f"Imagem pronta: {filename}")
            st.image(content, caption="Imagem selecionada", use_container_width=True)
        else:
            empty_state(
                "Nenhuma imagem selecionada",
                "Envie um arquivo ou escolha um exemplo para liberar a tradução.",
            )

section_title("Processamento", "Acompanhe a jornada da página até virar uma versão traduzida.")

if content is not None and filename is not None:
    current_job_key = make_job_key(content, filename, source_lang, translation_style, performance_mode, debug_enabled)
    st.caption("O app inicia automaticamente: detectar balões, ler textos, traduzir, limpar o original e inserir a versão em português.")

    cached_result = st.session_state.get("last_result")
    cached_key = st.session_state.get("last_job_key")
    if cached_result is not None and cached_key == current_job_key:
        status_pill("Resultado reaproveitado da sessão")
        render_result(cached_result, source_lang_label, performance_label)
    else:
        stage_state = make_stage_state()
        progress_state = {
            "active_stage_id": "load_image",
            "active_stage_label": "Preparando processamento",
            "progress_message": "Aquecendo o fluxo de tradução...",
            "latest_percent": 0,
        }

        with st.container(border=True):
            st.markdown("#### Progresso do processamento")
            header_slot = st.empty()
            progress_bar = st.progress(0)
            header_slot.markdown(
                render_progress_header(
                    progress_state["latest_percent"],
                    progress_state["active_stage_label"],
                    progress_state["progress_message"],
                ),
                unsafe_allow_html=True,
            )
            status_panel = st.status("Preparando a página...", expanded=True) if hasattr(st, "status") else None

            def update_progress(value: float, message: str, meta: dict[str, Any] | None = None) -> None:
                progress_state["latest_percent"] = int(max(0, min(100, round(float(value) * 100))))
                progress_state["active_stage_id"] = infer_stage_id(message, meta)
                progress_state["active_stage_label"] = clean_text((meta or {}).get("stage_label")) or next(
                    (step["title"] for step in PROCESS_STEPS if step["id"] == progress_state["active_stage_id"]),
                    "Processando",
                )
                progress_state["progress_message"] = clean_text(message).capitalize() or "Processando..."
                update_stage_state(
                    stage_state,
                    progress_state["active_stage_id"],
                    "done" if progress_state["latest_percent"] >= 100 else "running",
                )
                progress_bar.progress(progress_state["latest_percent"])
                header_slot.markdown(
                    render_progress_header(
                        progress_state["latest_percent"],
                        progress_state["active_stage_label"],
                        progress_state["progress_message"],
                    ),
                    unsafe_allow_html=True,
                )
                if status_panel is not None:
                    status_panel.update(
                        label=(
                            f"{progress_state['latest_percent']}% · "
                            f"{progress_state['active_stage_label']} · "
                            f"{progress_state['progress_message']}"
                        ),
                        state="complete" if progress_state["latest_percent"] >= 100 else "running",
                        expanded=True,
                    )

            try:
                with st.spinner("Analisando a página e preparando os modelos necessários..."):
                    result = process_uploaded_image(
                        filename,
                        content,
                        source_lang=source_lang,
                        translation_style=translation_style,
                        performance_mode=performance_mode,
                        debug_enabled=debug_enabled,
                        progress_callback=update_progress,
                    )
                st.session_state["last_result"] = result
                st.session_state["last_job_key"] = current_job_key
                mark_all_done(stage_state)
                progress_bar.progress(100)
                header_slot.markdown(
                    render_progress_header(100, "Resultado pronto", "Tradução finalizada com sucesso."),
                    unsafe_allow_html=True,
                )
                if status_panel is not None:
                    status_panel.update(label="100% · Resultado pronto", state="complete", expanded=False)
                show_completion_feedback(current_job_key)

                st.success("Tradução finalizada. A imagem está pronta para revisão e download.")
                render_result(result, source_lang_label, performance_label)

            except Exception as exc:
                if status_panel is not None:
                    status_panel.update(label="O processamento encontrou um erro", state="error", expanded=True)
                progress_bar.empty()
                header_slot.markdown(
                    render_progress_header(
                        progress_state["latest_percent"],
                        "Processamento interrompido",
                        "Algo falhou antes de finalizar a página.",
                    ),
                    unsafe_allow_html=True,
                )
                render_error(exc, stage_state, progress_state["active_stage_id"], traceback.format_exc())
else:
    empty_state(
        "Comece pela imagem",
        "Depois de selecionar uma página, a tradução automática começa aqui.",
    )
