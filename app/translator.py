from __future__ import annotations

import re
import threading
from pathlib import Path

from app.config import AppConfig
from app.utils import safe_text


PROVIDER_NAME = "small100_ct2"
MODEL_REPO = "entai2965/small100-ctranslate2"
VOCAB_FALLBACK_REPO = "alirezamsh/small100"


_LANG_ALIAS = {
    "jp": "ja",
    "japanese": "ja",
    "japones": "ja",
    "japonês": "ja",
    "en_us": "en",
    "en_gb": "en",
    "english": "en",
    "ingles": "en",
    "inglês": "en",
    "pt_br": "pt",
    "pt_pt": "pt",
    "portuguese": "pt",
    "portugues": "pt",
    "português": "pt",
    "zh_cn": "zh",
    "zh_tw": "zh",
    "zh_hans": "zh",
    "chinese": "zh",
    "chines": "zh",
    "chinês": "zh",
    "ko_kr": "ko",
    "korean": "ko",
    "ru_ru": "ru",
    "russian": "ru",
    "es_es": "es",
    "spanish": "es",
    "espanhol": "es",
    "auto": "auto",
}

SMALL100_LANGS = {
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs",
    "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa",
    "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv",
    "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv",
    "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns",
    "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk",
    "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl",
    "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu",
}


_FORBIDDEN_PREFIXES = (
    "translation:",
    "translated:",
    "portuguese:",
    "english:",
    "spanish:",
    "japanese:",
    "tradução:",
    "traducao:",
)

_SCRIPT_PATTERNS = {
    "ja": re.compile(r"[぀-ゟ゠-ヿ一-龯]"),
    "zh": re.compile(r"[一-鿿]"),
    "ko": re.compile(r"[가-힯]"),
    "ru": re.compile(r"[Ѐ-ӿ]"),
}


def normalize_lang_code(value, default: str = "") -> str:
    text = safe_text(value).lower().replace("-", "_")
    if not text:
        return default
    code = _LANG_ALIAS.get(text, text)
    if "_" in code:
        code = code.split("_", 1)[0]
    return code or default


def is_translation_valid(
    original: str,
    translated: str,
    target_lang: str,
) -> tuple[bool, str]:
    tgt_text = safe_text(translated)
    src_text = safe_text(original)

    if not tgt_text:
        return False, "empty"

    stripped = re.sub(r"[\s\.,;:!?\-—…\"'\(\)\[\]·]+", "", tgt_text)
    if not stripped:
        return False, "only_punctuation_or_space"

    lower = tgt_text.lower()
    for prefix in _FORBIDDEN_PREFIXES:
        if lower.startswith(prefix):
            return False, f"has_prefix:{prefix}"

    tgt_norm = normalize_lang_code(target_lang, "pt")

    if tgt_text == src_text:
        if tgt_norm in _SCRIPT_PATTERNS and _SCRIPT_PATTERNS[tgt_norm].search(src_text):
            return True, "same_text_already_in_target"
        return False, "same_as_source"

    if tgt_norm in _SCRIPT_PATTERNS:
        if not _SCRIPT_PATTERNS[tgt_norm].search(tgt_text):
            return False, f"missing_{tgt_norm}_script"
        return True, "ok"

    for src_code, pattern in _SCRIPT_PATTERNS.items():
        if pattern.search(tgt_text):
            return False, f"still_in_{src_code}_script"

    return True, "ok"


def normalize_punctuation(text: str, target_lang: str) -> str:
    clean = safe_text(text)
    if not clean:
        return ""

    clean = clean.replace("…", "...")
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"\s*\.\s*\.\s*\.", "...", clean)
    clean = re.sub(r"\.{4,}", "...", clean)
    clean = re.sub(r"!{3,}", "!!", clean)
    clean = re.sub(r"\?{3,}", "??", clean)
    clean = re.sub(r"\s+([!?]+)", r"\1", clean)
    clean = re.sub(r"\s+", " ", clean)
    return safe_text(clean)


def preserve_terminal_punctuation(source: str, translated: str, target_lang: str) -> str:
    clean_translated = safe_text(translated)
    if not clean_translated:
        return ""

    _, source_punctuation = _split_terminal_punctuation(source)
    if not source_punctuation:
        return clean_translated
    if re.search(r"([!?.]+|\?!|!\?)$", clean_translated):
        return clean_translated
    return normalize_punctuation(clean_translated + source_punctuation, target_lang)


def _split_terminal_punctuation(text: str) -> tuple[str, str]:
    clean = safe_text(text)
    if not clean:
        return "", ""

    for raw, mapped in (("！？", "?!"), ("？！", "?!"), ("。", "."), ("！", "!"), ("？", "?"), ("…", "...")):
        if clean.endswith(raw):
            return clean[: -len(raw)], mapped

    match = re.search(r"([!?.]+|\?!|!\?)$", clean)
    if not match:
        return clean, ""
    return clean[: match.start()], match.group(0)


class BaseTranslator:
    def translate(self, text: str) -> str:
        raise NotImplementedError

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]


class IdentityTranslator(BaseTranslator):
    def translate(self, text: str) -> str:
        return safe_text(text)

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [safe_text(text) for text in texts]


class Small100CT2Translator(BaseTranslator):
    _load_lock = threading.Lock()
    _shared_translator = None
    _shared_spm = None

    def __init__(self, config: AppConfig):
        self.config = config
        self.default_source = normalize_lang_code(getattr(config, "source_lang", "en"), "auto") or "auto"
        self.default_target = normalize_lang_code(getattr(config, "target_lang", "pt"), "pt") or "pt"
        if self.default_target not in SMALL100_LANGS:
            self.default_target = "pt"
        self.cache_enabled = bool(getattr(config, "translation_cache_enabled", True))
        self._beam_size = int(getattr(config, "translation_beam_size", 4) or 4)
        self._cache: dict[tuple[str, str, str], str] = {}
        print(
            f"[TRANSLATOR] provider={PROVIDER_NAME} model={MODEL_REPO} "
            f"source={self.default_source} target={self.default_target}"
        )

    @classmethod
    def _ensure_model(cls):
        if cls._shared_translator is not None:
            return cls._shared_translator, cls._shared_spm

        with cls._load_lock:
            if cls._shared_translator is not None:
                return cls._shared_translator, cls._shared_spm

            import ctranslate2
            import sentencepiece as spm_lib
            from huggingface_hub import snapshot_download

            print(f"[TRANSLATOR_LOAD] model={MODEL_REPO}")
            model_dir = Path(snapshot_download(repo_id=MODEL_REPO))
            spm_path = cls._find_spm(model_dir)
            if spm_path is None:
                fallback_dir = Path(
                    snapshot_download(
                        repo_id=VOCAB_FALLBACK_REPO,
                        allow_patterns=[
                            "sentencepiece.bpe.model",
                            "sentencepiece.model",
                            "spm.model",
                            "vocab.json",
                        ],
                    )
                )
                spm_path = cls._find_spm(fallback_dir)
            if spm_path is None:
                raise FileNotFoundError(
                    "SentencePiece model file não encontrado nem em "
                    f"{MODEL_REPO} nem em {VOCAB_FALLBACK_REPO}."
                )

            sp = spm_lib.SentencePieceProcessor()
            sp.load(str(spm_path))

            translator = ctranslate2.Translator(
                str(model_dir),
                device="cpu",
                compute_type="int8",
            )

            cls._shared_translator = translator
            cls._shared_spm = sp
            return translator, sp

    @staticmethod
    def _find_spm(directory: Path) -> Path | None:
        for name in ("sentencepiece.bpe.model", "sentencepiece.model", "spm.model"):
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def translate(self, text: str) -> str:
        return self.translate_text(text, self.default_source, self.default_target)

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [
            self.translate_text(t, self.default_source, self.default_target)
            for t in texts
        ]

    def translate_text(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "pt",
    ) -> str:
        src = normalize_lang_code(source_lang, self.default_source) or "auto"
        tgt = normalize_lang_code(target_lang, self.default_target) or "pt"
        if tgt not in SMALL100_LANGS:
            tgt = "pt"

        clean = safe_text(text)
        if not clean:
            return ""

        cache_key = (clean, src, tgt)
        if self.cache_enabled and cache_key in self._cache:
            print(f"[TRANSLATION_CACHE_HIT] source={src} target={tgt}")
            return self._cache[cache_key]

        translator, sp = self._ensure_model()

        pieces = sp.encode(clean, out_type=str)
        tgt_token = f"__{tgt}__"
        source_tokens = [tgt_token] + pieces + ["</s>"]

        results = translator.translate_batch(
            [source_tokens],
            beam_size=max(1, self._beam_size),
            max_decoding_length=256,
            replace_unknowns=True,
        )

        out_tokens = [
            t for t in results[0].hypotheses[0] if t != tgt_token and t != "</s>"
        ]

        try:
            output_text = sp.decode(out_tokens)
        except AttributeError:
            output_text = sp.decode_pieces(out_tokens)

        output_text = safe_text(output_text).strip()

        if self.cache_enabled:
            self._cache[cache_key] = output_text

        return output_text
