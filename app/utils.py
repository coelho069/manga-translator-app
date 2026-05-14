from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TRANSLATION_MODES = {
    "en_to_pt": {
        "label": "Inglês → Português",
        "source_lang": "en",
        "target_lang": "pt",
        "ocr_lang": "en",
    },
    "ja_to_en": {
        "label": "Japonês → Inglês",
        "source_lang": "ja",
        "target_lang": "en",
        "ocr_lang": "japan",
    },
}


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
    return " ".join(str(value).split())


def normalize_source_lang(value) -> str:
    lang = safe_text(value).lower().replace("_", "-")
    aliases = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "ingles": "en",
        "inglês": "en",
        "pt": "pt",
        "pt-br": "pt",
        "portugues": "pt",
        "português": "pt",
        "portuguese": "pt",
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-hans": "zh-CN",
        "cn": "zh-CN",
        "ch": "zh-CN",
        "chi": "zh-CN",
        "chinese": "zh-CN",
        "chines": "zh-CN",
        "chinês": "zh-CN",
        "chines simplificado": "zh-CN",
        "chinês simplificado": "zh-CN",
        "ja": "ja",
        "jp": "ja",
        "jpn": "ja",
        "japan": "ja",
        "japanese": "ja",
        "japones": "ja",
        "japonês": "ja",
    }
    return aliases.get(lang, "en")


def resolve_ocr_lang(value) -> str:
    normalized = normalize_source_lang(value)
    if normalized == "zh-CN":
        return "ch"
    if normalized == "ja":
        return "japan"
    return "en"


def resolve_translation_lang(value) -> str:
    return normalize_source_lang(value)


def resolve_translation_mode(value) -> str:
    raw = safe_text(value).lower().replace("-", "_")
    aliases = {
        "en_to_pt": "en_to_pt",
        "ingles_para_portugues": "en_to_pt",
        "inglês_para_português": "en_to_pt",
        "inglês_→_português": "en_to_pt",
        "english_to_portuguese": "en_to_pt",
        "ja_to_en": "ja_to_en",
        "japones_para_ingles": "ja_to_en",
        "japonês_para_inglês": "ja_to_en",
        "japonês_→_inglês": "ja_to_en",
        "japanese_to_english": "ja_to_en",
    }
    mode = aliases.get(raw, raw)
    if mode not in TRANSLATION_MODES:
        return "en_to_pt"
    return mode


def get_translation_mode_config(value) -> dict[str, str]:
    mode = resolve_translation_mode(value)
    config = TRANSLATION_MODES[mode]
    return {
        "mode": mode,
        "label": safe_text(config.get("label")),
        "source_lang": resolve_translation_lang(config.get("source_lang")),
        "target_lang": resolve_translation_lang(config.get("target_lang")),
        "ocr_lang": safe_text(config.get("ocr_lang")) or "en",
    }


def get_translation_mode_labels() -> dict[str, str]:
    return {mode: safe_text(config.get("label")) for mode, config in TRANSLATION_MODES.items()}


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
