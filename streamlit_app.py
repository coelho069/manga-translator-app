from __future__ import annotations

import traceback
from hashlib import sha256
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image

from app.backend import process_uploaded_image


EXAMPLES_DIR = Path("examples") / "bubbles"
LANG_OPTIONS = {"Inglês": "en", "Chinês": "zh-CN"}
STYLE_OPTIONS = {"Natural": "natural", "Literal": "literal"}
PERFORMANCE_OPTIONS = {"Equilibrado": "balanced", "Rápido": "fast", "Qualidade": "quality"}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0b0f19;
            --panel: rgba(20, 27, 43, 0.86);
            --panel-strong: rgba(27, 36, 57, 0.94);
            --panel-soft: rgba(255, 255, 255, 0.055);
            --stroke: rgba(255, 255, 255, 0.11);
            --stroke-strong: rgba(255, 255, 255, 0.18);
            --text: #f3f7ff;
            --muted: #9aa8c2;
            --accent: #7c5cff;
            --accent-2: #20d6a3;
            --danger: #ff6b8a;
            --shadow: 0 18px 55px rgba(0, 0, 0, 0.38);
            --radius: 18px;
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(124, 92, 255, 0.14) 0%, rgba(11, 15, 25, 0) 34%),
                linear-gradient(225deg, rgba(32, 214, 163, 0.10) 0%, rgba(11, 15, 25, 0) 36%),
                linear-gradient(180deg, #0b0f19 0%, #0d1322 48%, #0a0d15 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2.1rem;
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
            border-radius: 24px;
            padding: 30px 32px;
            background:
                linear-gradient(135deg, rgba(124, 92, 255, 0.20), rgba(32, 214, 163, 0.08)),
                rgba(18, 25, 40, 0.78);
            box-shadow: var(--shadow);
            animation: fadeUp 360ms ease-out both;
        }

        .eyebrow {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .app-hero h1 {
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.05;
            margin: 0 0 0.8rem 0;
        }

        .app-hero p {
            color: var(--muted);
            max-width: 780px;
            font-size: 1.04rem;
            line-height: 1.65;
            margin: 0;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }

        .badge {
            border: 1px solid var(--stroke);
            border-radius: 999px;
            color: #dbe5ff;
            background: rgba(255, 255, 255, 0.07);
            padding: 7px 12px;
            font-size: 0.84rem;
        }

        .section-title {
            margin: 2rem 0 0.8rem;
        }

        .section-title h2 {
            color: var(--text);
            font-size: 1.28rem;
            margin: 0 0 0.2rem;
        }

        .section-title p {
            color: var(--muted);
            margin: 0;
            line-height: 1.55;
        }

        .soft-card {
            border: 1px solid var(--stroke);
            border-radius: var(--radius);
            background: var(--panel);
            padding: 18px 18px;
            box-shadow: var(--shadow);
            animation: fadeUp 360ms ease-out both;
        }

        .empty-state {
            border: 1px dashed rgba(255, 255, 255, 0.20);
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
            border: 1px solid rgba(32, 214, 163, 0.28);
            color: #c8fff1;
            background: rgba(32, 214, 163, 0.10);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.9rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--accent-2);
            box-shadow: 0 0 18px rgba(32, 214, 163, 0.75);
            animation: breathe 1.3s ease-in-out infinite;
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
            border-color: rgba(124, 92, 255, 0.42);
        }

        .bubble-title {
            color: var(--text);
            font-weight: 700;
            margin-bottom: 8px;
        }

        .bubble-label {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-top: 8px;
        }

        .bubble-text {
            color: #dce6f8;
            line-height: 1.5;
            margin: 2px 0 8px;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border: 0 !important;
            border-radius: 14px !important;
            color: #ffffff !important;
            background: linear-gradient(135deg, #7c5cff, #20d6a3) !important;
            box-shadow: 0 12px 30px rgba(124, 92, 255, 0.28);
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
            font-weight: 800 !important;
            min-height: 46px;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-2px);
            filter: brightness(1.06);
            box-shadow: 0 16px 38px rgba(32, 214, 163, 0.22);
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(124, 92, 255, 0.40);
            border-radius: var(--radius);
            background: rgba(124, 92, 255, 0.06);
            padding: 10px;
            transition: border-color 160ms ease, background 160ms ease;
        }

        div[data-testid="stFileUploader"]:hover {
            border-color: rgba(32, 214, 163, 0.58);
            background: rgba(32, 214, 163, 0.06);
        }

        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, #7c5cff, #20d6a3) !important;
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
            50% { border-color: rgba(124, 92, 255, 0.45); }
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .app-hero {
                padding: 24px 20px;
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
            <div class="eyebrow">Local manga translation workspace</div>
            <h1>Manga Translator App</h1>
            <p>
                Traduza páginas de mangá em inglês ou chinês para português, mantendo o fluxo visual:
                detecção de balões, OCR, tradução, limpeza do texto original e renderização da versão final.
            </p>
            <div class="hero-badges">
                <span class="badge">YOLO segmentation</span>
                <span class="badge">PaddleOCR</span>
                <span class="badge">PT-BR natural</span>
                <span class="badge">Execução local</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text: str) -> None:
    st.markdown(
        status_pill_html(text),
        unsafe_allow_html=True,
    )


def status_pill_html(text: str) -> str:
    return f"""
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>{escape(str(text))}</span>
        </div>
        """


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bubble_info_card(bubble, source_label: str) -> None:
    source_text = escape(bubble.source_text or "(sem texto detectado)")
    translated_text = escape(bubble.translated_text or "(sem tradução)")
    source_label = escape(source_label)
    st.markdown(
        f"""
        <div class="bubble-row">
            <div class="bubble-title">Balão {bubble.id}</div>
            <div class="bubble-label">Original ({source_label})</div>
            <div class="bubble-text">{source_text}</div>
            <div class="bubble-label">Português</div>
            <div class="bubble-text">{translated_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def render_result(result, source_lang_label: str) -> None:
    timings = result.metadata.get("timings", {}) if result.metadata else {}
    total_time = timings.get("total")
    subtitle = "Compare a imagem original com a página traduzida."
    if total_time is not None:
        subtitle = f"{subtitle} Tempo total: {total_time:.2f}s."
    section_title("Resultado", subtitle)
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Original")
            st.image(Image.open(result.original_image_path), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("#### Traduzida")
            st.image(Image.open(result.translated_image_path), use_container_width=True)

    with open(result.translated_image_path, "rb") as image_file:
        st.download_button(
            "Baixar imagem traduzida",
            data=image_file,
            file_name=result.translated_image_path.name,
            mime="image/png",
            use_container_width=True,
        )

    if timings:
        with st.expander("Tempos por etapa"):
            st.json(timings)

    section_title("Textos detectados", "Revise o OCR e a tradução aplicada em cada balão.")
    if result.bubbles:
        for bubble in result.bubbles:
            bubble_info_card(bubble, source_lang_label)
    else:
        empty_state(
            "Nenhum balão detectado",
            "Confira se o modelo models/bubble_seg.pt é adequado para esse tipo de página.",
        )


st.set_page_config(page_title="Manga Translator App", page_icon="M", layout="wide")
inject_theme()
render_header()

example_files = sorted(EXAMPLES_DIR.glob("*.png")) if EXAMPLES_DIR.exists() else []

section_title("Entrada e configurações", "Escolha a imagem e ajuste idioma/estilo antes de iniciar o processamento.")
settings_col, preview_col = st.columns([0.92, 1.08], gap="large")

with settings_col:
    with st.container(border=True):
        st.markdown("#### Configurações")
        source_lang_label = st.selectbox("Idioma de entrada", list(LANG_OPTIONS.keys()))
        source_lang = LANG_OPTIONS[source_lang_label]

        translation_style_label = st.selectbox("Estilo de tradução", list(STYLE_OPTIONS.keys()))
        translation_style = STYLE_OPTIONS[translation_style_label]
        performance_label = st.selectbox("Modo de performance", list(PERFORMANCE_OPTIONS.keys()))
        performance_mode = PERFORMANCE_OPTIONS[performance_label]
        debug_enabled = st.toggle("Salvar debug visual", value=False)

        st.markdown("#### Imagem")
        use_example = st.toggle("Usar imagem de exemplo", disabled=not example_files)
        selected_example = None
        if use_example and example_files:
            selected_example = st.selectbox(
                "Imagem de exemplo",
                example_files,
                format_func=lambda path: path.name,
            )
        elif not example_files:
            st.caption("Gere exemplos com: python examples/create_bubble_examples.py")

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

section_title("Processamento", "Acompanhe as etapas do pipeline e veja o resultado lado a lado.")

if content is not None and filename is not None:
    current_job_key = make_job_key(content, filename, source_lang, translation_style, performance_mode, debug_enabled)
    action_col, hint_col = st.columns([0.32, 0.68], gap="large")
    with action_col:
        start = st.button("Iniciar tradução", type="primary", use_container_width=True)
    with hint_col:
        st.caption("O app vai detectar balões, ler o texto, traduzir, limpar o original e renderizar a versão final.")

    cached_result = st.session_state.get("last_result")
    cached_key = st.session_state.get("last_job_key")
    if cached_result is not None and cached_key == current_job_key and not start:
        status_pill("Resultado reaproveitado da sessão")
        render_result(cached_result, source_lang_label)

    if start:
        with st.container(border=True):
            st.markdown("#### Progresso")
            progress_bar = st.progress(0)
            status = st.empty()

            def update_progress(value: float, message: str) -> None:
                progress_bar.progress(int(value * 100))
                status.markdown(status_pill_html(str(message).capitalize()), unsafe_allow_html=True)

            try:
                if cached_result is not None and cached_key == current_job_key:
                    result = cached_result
                    progress_bar.progress(100)
                    status.success("Resultado reaproveitado da sessão.")
                else:
                    with st.spinner("Preparando a página e carregando os modelos necessários..."):
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

                progress_bar.progress(100)
                status.success("Tradução finalizada com sucesso.")

                render_result(result, source_lang_label)

            except Exception as exc:
                progress_bar.empty()
                status.empty()
                st.error(str(exc) or "Não foi possível processar a imagem.")
                with st.expander("Detalhes técnicos do erro"):
                    st.code(traceback.format_exc(), language="python")
else:
    empty_state(
        "Comece pela imagem",
        "Depois de selecionar uma página, o botão de tradução aparecerá aqui.",
    )
