from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st
from PIL import Image

from app.backend import process_uploaded_image


EXAMPLES_DIR = Path("examples") / "bubbles"


st.set_page_config(page_title="Manga Translator App", page_icon="M", layout="wide")

st.title("Manga Translator App")
st.write(
    "Envie uma pagina de manga em ingles ou chines para detectar baloes, ler o texto, traduzir para portugues "
    "e gerar uma nova imagem com a traducao aplicada."
)

source_lang_label = st.selectbox("Idioma de entrada", ["Inglês", "Chinês"])
source_lang = "zh-CN" if source_lang_label == "Chinês" else "en"
translation_style_label = st.selectbox("Estilo de traducao", ["Natural", "Literal"])
translation_style = "literal" if translation_style_label == "Literal" else "natural"

uploaded_file = st.file_uploader("Imagem PNG, JPG ou JPEG", type=["png", "jpg", "jpeg"])

example_files = sorted(EXAMPLES_DIR.glob("*.png")) if EXAMPLES_DIR.exists() else []
use_example = st.checkbox("Usar imagem de exemplo", disabled=not example_files)
selected_example = None

if use_example and example_files:
    selected_example = st.selectbox(
        "Imagem de exemplo",
        example_files,
        format_func=lambda path: path.name,
    )
elif not example_files:
    st.caption("Para habilitar exemplos, execute: python examples/create_bubble_examples.py")

content = None
filename = None

if use_example and selected_example is not None:
    content = selected_example.read_bytes()
    filename = selected_example.name
elif uploaded_file is not None:
    content = uploaded_file.getvalue()
    filename = uploaded_file.name

if content is not None and filename is not None:
    st.subheader("Pre-visualizacao")
    st.image(content, caption="Imagem original", use_container_width=True)

    if st.button("Iniciar traducao", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(value: float, message: str) -> None:
            progress_bar.progress(int(value * 100))
            status.write(message)

        try:
            result = process_uploaded_image(
                filename,
                content,
                source_lang=source_lang,
                translation_style=translation_style,
                progress_callback=update_progress,
            )
            progress_bar.progress(100)
            status.success("Traducao finalizada.")

            left, right = st.columns(2)
            with left:
                st.subheader("Original")
                st.image(Image.open(result.original_image_path), use_container_width=True)
            with right:
                st.subheader("Traduzida")
                st.image(Image.open(result.translated_image_path), use_container_width=True)

            with open(result.translated_image_path, "rb") as image_file:
                st.download_button(
                    "Baixar imagem traduzida",
                    data=image_file,
                    file_name=result.translated_image_path.name,
                    mime="image/png",
                )

            st.subheader("Textos detectados")
            if result.bubbles:
                for bubble in result.bubbles:
                    with st.container(border=True):
                        st.markdown(f"**Balao {bubble.id}**")
                        st.write(f"**Original ({source_lang_label}):**", bubble.source_text or "(sem texto detectado)")
                        st.write("**Traducao em portugues:**", bubble.translated_text or "(sem traducao)")
            else:
                st.info("Nenhum balao foi detectado na imagem.")

        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.error(str(exc) or "Ocorreu um erro ao processar a imagem.")
            with st.expander("Detalhes tecnicos do erro"):
                st.code(traceback.format_exc(), language="python")
else:
    st.info("Envie uma imagem ou selecione um exemplo para comecar.")
