from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from app.config import AppConfig
from app.utils import safe_text


PROVIDER_NAME = "m2m100"
MODEL_REPO = "facebook/m2m100_418M"
VOCAB_FALLBACK_REPO = "alirezamsh/small100"


_LANG_ALIAS = {
    "jp": "ja",
    "japanese": "ja",
    "japones": "ja",
    "japonês": "ja",
    "ja_jp": "ja",
    "en_us": "en",
    "en-us": "en",
    "en_gb": "en",
    "en-gb": "en",
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
    "mandarin": "zh",
    "ko_kr": "ko",
    "korean": "ko",
    "coreano": "ko",
    "ru_ru": "ru",
    "russian": "ru",
    "russo": "ru",
    "es_es": "es",
    "spanish": "es",
    "espanhol": "es",
    "auto": "auto",
}

MBART_LANG_CODES = {
    "en": "en_XX",
    "ja": "ja_XX",
    "pt": "pt_XX",
    "zh": "zh_CN",
    "ko": "ko_KR",
    "ru": "ru_RU",
    "es": "es_XX",
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


def validate_translation_output(original: str, translated: str, source_lang: str, target_lang: str) -> tuple[bool, str]:
    valid, reason = is_translation_valid(original, translated, target_lang)
    if not valid:
        return valid, reason

    src = normalize_lang_code(source_lang, "auto")
    tgt = normalize_lang_code(target_lang, "pt")
    src_text = safe_text(original)
    out_text = safe_text(translated)
    if src != "auto" and src != tgt and out_text == src_text:
        return False, "same_as_source"
    if src != "auto" and src != tgt and src in _SCRIPT_PATTERNS and _SCRIPT_PATTERNS[src].search(out_text):
        return False, f"still_in_{src}_script"
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


def build_translator(config: AppConfig) -> BaseTranslator:
    provider = safe_text(getattr(config, "translation_provider", PROVIDER_NAME)).lower()
    if provider in {"m2m100", "mbart", "small100", "transformers_seq2seq", "seq2seq", ""}:
        return TransformersSeq2SeqTranslator(config)
    if provider in {"small100_ct2", "ct2"}:
        return Small100CT2Translator(config)
    print(f"[TRANSLATOR] provider={provider} desconhecido; usando m2m100")
    return TransformersSeq2SeqTranslator(config)


class TransformersSeq2SeqTranslator(BaseTranslator):
    _load_lock = threading.Lock()
    _shared_model = None
    _shared_tokenizer = None
    _shared_repo = ""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model_repo = safe_text(getattr(config, "translation_model", MODEL_REPO)) or MODEL_REPO
        self.model_kind = self._model_kind(self.model_repo, getattr(config, "translation_provider", PROVIDER_NAME))
        self.default_source = normalize_lang_code(getattr(config, "source_lang", "auto"), "auto") or "auto"
        self.default_target = normalize_lang_code(getattr(config, "target_lang", "pt"), "pt") or "pt"
        self.cache_enabled = bool(getattr(config, "translation_cache_enabled", True))
        self._cache: dict[tuple[str, str, str], str] = {}
        print(
            f"[TRANSLATOR] model={self.model_repo} source={self.default_source} "
            f"target={self.default_target}"
        )

    @staticmethod
    def _model_kind(model_repo: str, provider: str) -> str:
        text = f"{provider} {model_repo}".lower()
        if "mbart" in text:
            return "mbart"
        if "small100" in text:
            return "small100"
        return "m2m100"

    @classmethod
    def _ensure_model(cls, repo_id: str):
        if cls._shared_model is not None and cls._shared_tokenizer is not None and cls._shared_repo == repo_id:
            return cls._shared_model, cls._shared_tokenizer

        with cls._load_lock:
            if cls._shared_model is not None and cls._shared_tokenizer is not None and cls._shared_repo == repo_id:
                return cls._shared_model, cls._shared_tokenizer

            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            print(f"[TRANSLATOR_LOAD] model={repo_id}")
            tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token or None)
            model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, token=hf_token or None)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            cls._shared_model = model
            cls._shared_tokenizer = tokenizer
            cls._shared_repo = repo_id
            print(f"[TRANSLATOR_LOADED] model={repo_id} device={device}")
            return model, tokenizer

    def translate(self, text: str) -> str:
        return self.translate_text(text, self.default_source, self.default_target)

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [self.translate_text(text, self.default_source, self.default_target) for text in texts]

    def translate_text(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "pt",
    ) -> str:
        clean = safe_text(text)
        if not clean:
            raise ValueError("empty_translation_input")

        src = normalize_lang_code(source_lang, self.default_source)
        if src == "auto":
            src = self._detect_or_default_source(clean)
        tgt = normalize_lang_code(target_lang, self.default_target) or "pt"
        print(f"[TRANSLATOR_LANG] normalized_source={src} normalized_target={tgt}")

        cache_key = (clean, src, tgt)
        if self.cache_enabled and cache_key in self._cache:
            print(f"[TRANSLATION_CACHE_HIT] source={src} target={tgt}")
            return self._cache[cache_key]

        model, tokenizer = self._ensure_model(self.model_repo)
        src_code, tgt_code = self._tokenizer_lang_codes(src, tgt)

        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_code

        print(f"[TRANSLATOR_TOKENIZER] src_lang={src_code} target_lang={tgt_code}")
        inputs = tokenizer(clean, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        forced_bos_token_id = self._forced_bos_token_id(tokenizer, tgt_code)
        print(f"[TRANSLATOR_GENERATE] forced_bos_token_id={forced_bos_token_id}")

        generate_kwargs = {
            **inputs,
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
        }
        if forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

        import torch

        try:
            with torch.inference_mode():
                generated = model.generate(**generate_kwargs)
            translated = tokenizer.decode(generated[0], skip_special_tokens=True)
            translated = self._clean_output(translated)
            valid, reason = validate_translation_output(clean, translated, src, tgt)
            print(f"[TRANSLATION_VALIDATE] valid={valid} reason={reason}")
            if not valid:
                raise ValueError(reason)
        except Exception as exc:
            print(f"[TRANSLATION_ERROR] error={exc}")
            raise

        if self.cache_enabled:
            self._cache[cache_key] = translated
        return translated

    def _detect_or_default_source(self, text: str) -> str:
        for code, pattern in _SCRIPT_PATTERNS.items():
            if pattern.search(text):
                return code
        default = normalize_lang_code(os.getenv("DEFAULT_SOURCE_LANG", self.default_source), "en")
        return "en" if default == "auto" else default

    def _tokenizer_lang_codes(self, source_lang: str, target_lang: str) -> tuple[str, str]:
        if self.model_kind == "mbart":
            return MBART_LANG_CODES.get(source_lang, source_lang), MBART_LANG_CODES.get(target_lang, target_lang)
        return source_lang, target_lang

    @staticmethod
    def _forced_bos_token_id(tokenizer, target_code: str):
        if hasattr(tokenizer, "get_lang_id"):
            return tokenizer.get_lang_id(target_code)
        lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict):
            return lang_code_to_id.get(target_code)
        if callable(lang_code_to_id):
            return lang_code_to_id(target_code)
        return None

    @staticmethod
    def _clean_output(text: str) -> str:
        clean = safe_text(text)
        clean = re.sub(
            r"^(translation|english|portuguese|spanish|japanese|tradu[cç][aã]o)\s*:\s*",
            "",
            clean,
            flags=re.I,
        )
        return safe_text(clean.strip(" \"'`"))


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
