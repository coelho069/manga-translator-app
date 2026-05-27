from __future__ import annotations

from pathlib import Path
from typing import Iterator

from app.utils import ensure_dir


def _require_fitz():
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF nao instalado. Instale com: python -m pip install pymupdf"
        ) from exc
    return fitz


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    if pdf_bytes is None or len(pdf_bytes) == 0:
        return 0

    fitz = _require_fitz()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return int(doc.page_count)


def iter_pdf_pages(pdf_bytes: bytes, output_dir: Path, dpi: int = 200) -> Iterator[tuple[int, int, Path]]:
    if pdf_bytes is None or len(pdf_bytes) == 0:
        return

    ensure_dir(output_dir)
    fitz = _require_fitz()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = int(doc.page_count)
        zoom = float(dpi) / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(total_pages):
            page = None
            pix = None
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                page_path = Path(output_dir) / f"page_{page_index + 1:04d}.png"
                pix.save(str(page_path))
                yield page_index + 1, total_pages, page_path
            finally:
                pix = None
                page = None
    finally:
        doc.close()
