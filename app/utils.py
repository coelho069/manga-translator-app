from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_image_cv2(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nao foi possivel ler a imagem: {path}")
    return image


def write_image_cv2(path: Path | str, image: np.ndarray) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise ValueError(f"Nao foi possivel salvar a imagem: {output_path}")
    return output_path


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def safe_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def sanitize_filename(filename: str) -> str:
    name = Path(safe_text(filename) or "image").name
    stem = Path(name).stem or "image"
    suffix = Path(name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
    return f"{clean_stem}{suffix}"


def make_job_dir(base_dir: Path | str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    return ensure_dir(Path(base_dir) / f"job_{timestamp}_{job_id}")

