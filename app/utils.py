from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TRANSLATION_FLOWS = {
    "auto_to_en": {
        "label": "Auto → Inglês",
        "source_lang": "auto",
        "target_lang": "en",
        "ocr_lang": "japan",
    },
    "auto_to_pt": {
        "label": "Auto → Português",
        "source_lang": "auto",
        "target_lang": "pt",
        "ocr_lang": "japan",
    },
    "ja_to_en": {
        "label": "Japonês → Inglês",
        "source_lang": "ja",
        "target_lang": "en",
        "ocr_lang": "japan",
    },
    "ja_to_pt": {
        "label": "Japonês → Português",
        "source_lang": "ja",
        "target_lang": "pt",
        "ocr_lang": "japan",
    },
    "en_to_pt": {
        "label": "Inglês → Português",
        "source_lang": "en",
        "target_lang": "pt",
        "ocr_lang": "en",
    },
    "zh_to_en": {
        "label": "Chinês → Inglês",
        "source_lang": "zh",
        "target_lang": "en",
        "ocr_lang": "ch",
    },
    "zh_to_pt": {
        "label": "Chinês → Português",
        "source_lang": "zh",
        "target_lang": "pt",
        "ocr_lang": "ch",
    },
    "ko_to_en": {
        "label": "Coreano → Inglês",
        "source_lang": "ko",
        "target_lang": "en",
        "ocr_lang": "korean",
    },
    "ko_to_pt": {
        "label": "Coreano → Português",
        "source_lang": "ko",
        "target_lang": "pt",
        "ocr_lang": "korean",
    },
    "ru_to_en": {
        "label": "Russo → Inglês",
        "source_lang": "ru",
        "target_lang": "en",
        "ocr_lang": "ru",
    },
    "ru_to_pt": {
        "label": "Russo → Português",
        "source_lang": "ru",
        "target_lang": "pt",
        "ocr_lang": "ru",
    },
    "es_to_en": {
        "label": "Espanhol → Inglês",
        "source_lang": "es",
        "target_lang": "en",
        "ocr_lang": "en",
    },
    "es_to_pt": {
        "label": "Espanhol → Português",
        "source_lang": "es",
        "target_lang": "pt",
        "ocr_lang": "en",
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
    try:
        max_side = int(os.getenv("MAX_INPUT_IMAGE_SIDE", "0") or "0")
    except ValueError:
        max_side = 0
    if max_side > 0:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[RESIZE] input downscaled from {w}x{h} to {new_w}x{new_h} (max_side={max_side})")
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
    # Delegate to the multilingual normalizer so OCR / UI / API share one map.
    from app.translator import normalize_lang_code

    return normalize_lang_code(value, default="auto")


def resolve_ocr_lang(value) -> str:
    normalized = normalize_source_lang(value)
    ocr_map = {
        "zh": "ch",
        "ja": "japan",
        "ko": "korean",
        "ru": "ru",
        "ar": "arabic",
        "en": "en",
        "pt": "pt",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
    }
    if normalized in ocr_map:
        return ocr_map[normalized]
    if normalized == "auto":
        return "japan"
    return "en"


def resolve_translation_lang(value) -> str:
    return normalize_source_lang(value)


def resolve_translation_flow(value) -> str:
    raw = safe_text(value).lower().replace("-", "_")
    aliases = {
        "japones_para_ingles": "ja_to_en",
        "japonês_para_inglês": "ja_to_en",
        "japonês_→_inglês": "ja_to_en",
        "japanese_to_english": "ja_to_en",
        "japones_para_portugues": "ja_to_pt",
        "japonês_para_português": "ja_to_pt",
        "japanese_to_portuguese": "ja_to_pt",
        "ch_to_en": "zh_to_en",
        "chines_para_ingles": "zh_to_en",
        "chinese_to_english": "zh_to_en",
        "ingles_para_portugues": "en_to_pt",
        "english_to_portuguese": "en_to_pt",
        "coreano_para_ingles": "ko_to_en",
        "korean_to_english": "ko_to_en",
        "russo_para_portugues": "ru_to_pt",
        "russian_to_portuguese": "ru_to_pt",
        "espanhol_para_ingles": "es_to_en",
        "spanish_to_english": "es_to_en",
        "auto_to_en": "auto_to_en",
        "auto_to_pt": "auto_to_pt",
    }
    mode = aliases.get(raw, raw)
    if mode not in TRANSLATION_FLOWS:
        # Default to the env-configured fallback when nothing matches.
        import os

        default_flow = os.getenv("DEFAULT_TRANSLATION_FLOW", "auto_to_en")
        return default_flow if default_flow in TRANSLATION_FLOWS else "auto_to_en"
    return mode


def get_translation_flow_config(value) -> dict[str, str]:
    mode = resolve_translation_flow(value)
    config = TRANSLATION_FLOWS[mode]
    return {
        "mode": mode,
        "label": safe_text(config.get("label")),
        "source_lang": resolve_translation_lang(config.get("source_lang")),
        "target_lang": resolve_translation_lang(config.get("target_lang")),
        "ocr_lang": safe_text(config.get("ocr_lang")) or "en",
    }


def get_translation_flow_labels() -> dict[str, str]:
    return {mode: safe_text(config.get("label")) for mode, config in TRANSLATION_FLOWS.items()}


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
