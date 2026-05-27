"""Multilingual translation engine.

Supports many source/target languages through a single entry point
:func:`translate_text`. The default provider is ``multilingual`` and loads a
seq2seq model (M2M100 / SMALL100 / CT2) once and reuses it for every balloon.

Environment variables
---------------------

``TRANSLATION_PROVIDER`` (default ``multilingual``)
    ``multilingual``: M2M100/SMALL100 family (HF transformers).
    ``ctranslate2``:  SMALL100/M2M100 models served via ``ctranslate2``
                       (requires the optional ``ctranslate2`` package).
    ``opus_mt``:      Legacy Helsinki-NLP/opus-mt-* models. Only used as an
                       explicit ja→en fallback by default.

``TRANSLATION_MODEL`` (default ``facebook/m2m100_418M``)
    Hugging Face model id (or local CT2 dir for ``ctranslate2``).

``DEFAULT_SOURCE_LANG`` (default ``auto``)
``DEFAULT_TARGET_LANG`` (default ``en``)
"""

from __future__ import annotations

import os
import re
import string
import threading
from dataclasses import dataclass

from app.hf_auth import hf_auth_help_message, hf_auth_kwargs, is_hf_auth_error

# --- Defaults -----------------------------------------------------------------

DEFAULT_PROVIDER = "multilingual"
DEFAULT_MODEL = "facebook/m2m100_418M"
DEFAULT_SOURCE_LANG = "auto"
DEFAULT_TARGET_LANG = "en"

# Fallback specifically for the ja→en pair when the multilingual model fails.
JA_EN_FALLBACK_MODEL = "Helsinki-NLP/opus-mt-ja-en"

# --- Script detection regexes -------------------------------------------------

_JAPANESE_RE = re.compile(r"[぀-ゟ゠-ヿ]")  # hiragana / katakana
_CJK_UNIFIED_RE = re.compile(r"[㐀-䶿一-鿿]")  # han characters
_HANGUL_RE = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_HEBREW_RE = re.compile(r"[֐-׿]")
_THAI_RE = re.compile(r"[฀-๿]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")

# Languages whose script makes a "still in source" check meaningful — we can
# only tell pt vs en apart by content, not by script, so Latin-script source
# languages are excluded from that validation.
_NON_LATIN_LANGS = frozenset({"ja", "zh", "ko", "ru", "ar", "he", "th", "hi"})

# Portuguese-specific diacritics that don't occur in English text.
_PT_DIACRITIC_RE = re.compile(r"[ãõçÃÕÇáàâéêíóôúüÁÀÂÉÊÍÓÔÚÜ]")
# Frequent short Portuguese function words / endings.
_PT_TOKEN_RE = re.compile(
    r"\b("
    r"de|do|da|dos|das|o|a|os|as|um|uma|uns|umas|que|não|nao|sim|"
    r"para|pra|por|com|sem|mas|ou|se|já|ja|também|tambem|muito|"
    r"você|voce|eu|ele|ela|eles|elas|nós|nos|isso|isto|aqui|ali|"
    r"está|esta|estou|sou|sao|são|tem|têm|foi|ser|ter|vai|vamos|"
    r"então|entao|agora|depois|antes|sempre|nunca"
    r")\b",
    re.IGNORECASE,
)
# Frequent short English function words. Used only to break ties.
_EN_TOKEN_RE = re.compile(
    r"\b("
    r"the|a|an|is|are|am|was|were|be|been|being|"
    r"i|you|he|she|it|we|they|me|him|her|us|them|"
    r"and|or|but|if|then|so|because|while|of|to|in|on|at|for|with|"
    r"from|by|about|as|this|that|these|those|here|there|"
    r"do|does|did|don't|doesn't|didn't|"
    r"have|has|had|will|would|can|could|should|may|might|"
    r"what|when|where|why|how|who|which|"
    r"hello|hi|hey|please|thanks|sorry|yes|no|okay|ok"
    r")\b",
    re.IGNORECASE,
)

# --- Language code aliases ---------------------------------------------------
# Normalize a wide variety of inputs (jp/japanese/portuguese/zh-CN/...) into
# the short ISO 639-1 codes the multilingual models expect.

_LANG_ALIASES: dict[str, str] = {
    "auto": "auto",
    "any": "auto",
    "detect": "auto",
    # English
    "en": "en", "eng": "en", "en-us": "en", "en-gb": "en",
    "english": "en", "ingles": "en", "inglês": "en", "inglesa": "en",
    # Portuguese
    "pt": "pt", "pt-br": "pt", "pt-pt": "pt", "ptbr": "pt", "ptpt": "pt",
    "por": "pt", "portuguese": "pt", "portugues": "pt", "português": "pt",
    "brazilian portuguese": "pt",
    # Spanish
    "es": "es", "spa": "es", "es-es": "es", "es-mx": "es",
    "spanish": "es", "espanhol": "es", "español": "es", "espanol": "es",
    # French
    "fr": "fr", "fra": "fr", "fre": "fr", "fr-fr": "fr",
    "french": "fr", "francais": "fr", "français": "fr", "frances": "fr", "francês": "fr",
    # German
    "de": "de", "deu": "de", "ger": "de", "de-de": "de",
    "german": "de", "alemao": "de", "alemão": "de",
    # Italian
    "it": "it", "ita": "it",
    "italian": "it", "italiano": "it",
    # Japanese
    "ja": "ja", "jp": "ja", "jpn": "ja", "japan": "ja",
    "japanese": "ja", "japones": "ja", "japonês": "ja",
    # Chinese (simplified)
    "zh": "zh", "zh-cn": "zh", "zh-hans": "zh", "cn": "zh", "chi": "zh",
    "chinese": "zh", "chines": "zh", "chinês": "zh",
    "chines simplificado": "zh", "chinês simplificado": "zh",
    # Korean
    "ko": "ko", "kor": "ko", "kr": "ko",
    "korean": "ko", "coreano": "ko",
    # Russian
    "ru": "ru", "rus": "ru",
    "russian": "ru", "russo": "ru",
    # Arabic
    "ar": "ar", "ara": "ar",
    "arabic": "ar", "arabe": "ar", "árabe": "ar",
    # Hebrew
    "he": "he", "heb": "he", "iw": "he",
    "hebrew": "he", "hebraico": "he",
    # Thai
    "th": "th", "tha": "th",
    "thai": "th", "tailandes": "th", "tailandês": "th",
    # Vietnamese
    "vi": "vi", "vie": "vi",
    "vietnamese": "vi", "vietnamita": "vi",
    # Hindi
    "hi": "hi", "hin": "hi",
    "hindi": "hi",
    # Turkish
    "tr": "tr", "tur": "tr",
    "turkish": "tr", "turco": "tr",
    # Dutch
    "nl": "nl", "nld": "nl", "dut": "nl",
    "dutch": "nl", "holandes": "nl", "holandês": "nl",
    # Polish
    "pl": "pl", "pol": "pl",
    "polish": "pl", "polones": "pl", "polonês": "pl",
}


def normalize_lang_code(value, *, default: str = "auto") -> str:
    """Normalize free-form language input into a canonical short code."""
    cleaned = safe_text(value).lower().strip().replace("_", "-")
    if not cleaned:
        return default
    if cleaned in _LANG_ALIASES:
        return _LANG_ALIASES[cleaned]
    # Take the first segment ("pt-br" → "pt") and try again.
    head = cleaned.split("-", 1)[0]
    if head in _LANG_ALIASES:
        return _LANG_ALIASES[head]
    return default


# --- Errors / settings -------------------------------------------------------


class TranslationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranslationSettings:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    source_lang: str = DEFAULT_SOURCE_LANG
    target_lang: str = DEFAULT_TARGET_LANG


# --- Public helpers ----------------------------------------------------------


def safe_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def get_translation_settings(config=None) -> TranslationSettings:
    provider = safe_text(os.getenv("TRANSLATION_PROVIDER")).lower()
    if not provider and config is not None:
        provider = safe_text(getattr(config, "translation_provider", ""))
    provider = provider or DEFAULT_PROVIDER
    if provider in {"local_hf", "hf"}:
        provider = "multilingual"

    model = safe_text(os.getenv("TRANSLATION_MODEL"))
    if not model:
        model = safe_text(os.getenv("HF_TRANSLATION_MODEL"))
    if not model and config is not None:
        model = safe_text(getattr(config, "hf_translation_model", ""))
    model = model or DEFAULT_MODEL

    source_lang = safe_text(os.getenv("DEFAULT_SOURCE_LANG"))
    if not source_lang:
        source_lang = safe_text(os.getenv("SOURCE_LANG"))
    if not source_lang and config is not None:
        source_lang = safe_text(getattr(config, "source_lang", ""))
    source_lang = normalize_lang_code(source_lang or DEFAULT_SOURCE_LANG, default=DEFAULT_SOURCE_LANG)

    target_lang = safe_text(os.getenv("DEFAULT_TARGET_LANG"))
    if not target_lang:
        target_lang = safe_text(os.getenv("TARGET_LANG"))
    if not target_lang and config is not None:
        target_lang = safe_text(getattr(config, "target_lang", ""))
    target_lang = normalize_lang_code(target_lang or DEFAULT_TARGET_LANG, default=DEFAULT_TARGET_LANG)

    return TranslationSettings(provider=provider, model=model, source_lang=source_lang, target_lang=target_lang)


# --- Translator class --------------------------------------------------------


class BaseTranslator:
    def translate(
        self,
        text: str,
        balloon_id: int | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        context: list[str] | None = None,
    ) -> str:
        raise NotImplementedError

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [self.translate(text) for text in texts]


class LocalHFTranslator(BaseTranslator):
    """Multilingual translator backed by an HF seq2seq model.

    The model is loaded lazily on first use and reused for every balloon.
    """

    def __init__(self, config=None):
        self.config = config
        self.settings = get_translation_settings(config)
        print(
            f"[TRANSLATOR] provider={self.settings.provider} model={self.settings.model} "
            f"source={self.settings.source_lang} target={self.settings.target_lang}"
        )

    def translate(
        self,
        text: str,
        balloon_id: int | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        context: list[str] | None = None,
    ) -> str:
        src = normalize_lang_code(source_lang or self.settings.source_lang, default=DEFAULT_SOURCE_LANG)
        tgt = normalize_lang_code(target_lang or self.settings.target_lang, default=DEFAULT_TARGET_LANG)
        return translate_text(
            text,
            source_lang=src,
            target_lang=tgt,
            config=self.config,
            balloon_id=balloon_id,
            context=context,
        )


# Aliases for older callers.
ConfigurableApiTranslator = LocalHFTranslator
GoogleTextTranslator = LocalHFTranslator


# --- Runtime cache -----------------------------------------------------------


@dataclass
class _Seq2SeqRuntime:
    tokenizer: object
    model: object
    torch: object
    device: object
    family: str  # "m2m100" | "small100" | "marian" | "ct2"
    model_id: str


_runtime_lock = threading.Lock()
_runtimes: dict[str, _Seq2SeqRuntime] = {}
_translation_cache: dict[tuple[str, str, str, str], str] = {}
_translation_cache_lock = threading.Lock()
_translation_cache_max_items = 4096

_debug_lock = threading.Lock()
_last_debug: dict[str, str] = {"raw": "", "decoded": ""}


def get_last_translation_debug() -> dict[str, str]:
    with _debug_lock:
        return {"raw": _last_debug.get("raw", ""), "decoded": _last_debug.get("decoded", "")}


# --- Public entry point ------------------------------------------------------


def translate_text(
    text: str,
    source_lang: str = DEFAULT_SOURCE_LANG,
    target_lang: str = DEFAULT_TARGET_LANG,
    *,
    config=None,
    balloon_id: int | None = None,
    context: list[str] | None = None,
) -> str:
    """Translate ``text`` from ``source_lang`` to ``target_lang``.

    ``source_lang="auto"`` triggers a lightweight script-based detection.
    Languages are normalized so callers may pass ``"jp"``, ``"japanese"``,
    ``"pt-BR"``, ``"english"`` etc. ``context`` is a short list of the
    surrounding bubble OCR texts on the same page — they are NOT merged into
    the model input (would mix dialogues) but are forwarded to the optional
    LLM refiner so it can pick a coherent tone.
    """
    raw = safe_text(text)
    if not raw:
        raise TranslationError("texto original vazio")

    settings = get_translation_settings(config)
    src = normalize_lang_code(source_lang, default=settings.source_lang)
    tgt = normalize_lang_code(target_lang, default=settings.target_lang)
    if src == "auto":
        src = detect_language(raw, fallback=settings.source_lang if settings.source_lang != "auto" else "en")
    if tgt == "auto":
        tgt = settings.target_lang if settings.target_lang != "auto" else DEFAULT_TARGET_LANG

    balloon_tag = balloon_id if balloon_id is not None else 0
    print(f"[TRANSLATION_QUALITY] start balloon={balloon_tag} source={src} target={tgt}")
    print(f"[TRANSLATION] start balloon={balloon_tag} source={src} target={tgt}")

    # ---- OCR clean-up before the model sees it ------------------------------
    original = _normalize_ocr_text(raw)
    if raw != original:
        print(f"[TRANSLATION_CLEAN_INPUT] balloon={balloon_tag} before=\"{raw}\" after=\"{original}\"")
    if not original:
        raise TranslationError("texto OCR vazio apos limpeza")
    print(f"[TRANSLATION_INPUT] balloon={balloon_tag} text=\"{original}\"")

    ctx_list = [safe_text(c) for c in (context or []) if safe_text(c)]
    if ctx_list:
        print(f"[TRANSLATION_CONTEXT] balloon={balloon_tag} context_count={len(ctx_list)}")

    # If the source already matches the target and the text contains nothing
    # from the source's script, treat the original as the translation.
    if src == tgt and not _text_is_in_language(original, src):
        print(f"[TRANSLATION_QUALITY_VALIDATE] balloon={balloon_tag} valid=true reason=same_lang_passthrough")
        print(f"[TRANSLATION_CHOSEN] balloon={balloon_tag} text=\"{original}\"")
        return original

    cached = _get_cached(original, settings, src, tgt)
    if cached is not None:
        print(f"[TRANSLATION_CHOSEN] balloon={balloon_tag} text=\"{cached}\"")
        return cached

    last_error: Exception | None = None
    candidates: list[tuple[str, str]] = [(settings.provider, settings.model)]
    if src == "ja" and tgt == "en" and settings.model != JA_EN_FALLBACK_MODEL:
        candidates.append(("opus_mt", JA_EN_FALLBACK_MODEL))

    valid_candidates: list[tuple[str, str]] = []  # (provider_label, text)

    for provider, model_id in candidates:
        for attempt_label, attempt_text in (("", original), ("_retry", _normalize_ocr_text(original))):
            if attempt_label == "_retry" and attempt_text == original:
                continue
            try:
                cand = _translate_with(
                    text=attempt_text,
                    source_lang=src,
                    target_lang=tgt,
                    provider=provider,
                    model_id=model_id,
                )
            except Exception as exc:
                last_error = exc
                print(f"[TRANSLATION_QUALITY_ERROR] balloon={balloon_tag} error={exc}")
                print(f"[TRANSLATION_ERROR] balloon={balloon_tag} error={exc}")
                print(f"[TRANSLATION_FALLBACK] balloon={balloon_tag} from={provider}:{model_id} to=next reason={exc}")
                break  # don't bother retrying a broken provider with retry text

            cleaned = _clean_translation_output(cand, tgt)
            label = f"{provider}{attempt_label}"
            print(f"[TRANSLATION_CANDIDATE] balloon={balloon_tag} provider={label} text=\"{cleaned}\"")
            valid, reason = validate_translation_for_target(attempt_text, cleaned, src, tgt)
            print(f"[TRANSLATION_QUALITY_VALIDATE] balloon={balloon_tag} valid={str(valid).lower()} reason={reason}")
            print(f"[TRANSLATION_VALIDATE] balloon={balloon_tag} valid={str(valid).lower()} reason={reason}")
            if valid:
                valid_candidates.append((label, cleaned))
                break  # first valid attempt for this provider is enough

        if valid_candidates:
            # Got at least one valid candidate for this provider — stop unless
            # the caller wanted to harvest fallbacks for scoring; we already
            # pick the best below.
            break

        print(f"[TRANSLATION_FALLBACK] balloon={balloon_tag} from={provider}:{model_id} to=next reason=invalid_or_error")

    if not valid_candidates:
        error = last_error or TranslationError(f"nao foi possivel traduzir {src}->{tgt}")
        print(f"[TRANSLATION_QUALITY_ERROR] balloon={balloon_tag} error={error}")
        if isinstance(error, TranslationError):
            raise error
        raise TranslationError(str(error))

    # Try the LLM refiner (if configured) and add its output as another
    # candidate. The refiner runs on the best primary candidate so far.
    primary_label, primary_text = max(valid_candidates, key=lambda c: _score_candidate(c[1], src, tgt))
    refined = _refine_with_llm(
        original=original,
        initial=primary_text,
        source_lang=src,
        target_lang=tgt,
        context=ctx_list,
        balloon_id=balloon_tag,
    )
    if refined:
        cleaned_ref = _clean_translation_output(refined, tgt)
        print(f"[TRANSLATION_CANDIDATE] balloon={balloon_tag} provider=refiner text=\"{cleaned_ref}\"")
        valid_r, reason_r = validate_translation_for_target(original, cleaned_ref, src, tgt)
        print(f"[TRANSLATION_QUALITY_VALIDATE] balloon={balloon_tag} valid={str(valid_r).lower()} reason={reason_r}")
        print(f"[TRANSLATION_VALIDATE] balloon={balloon_tag} valid={str(valid_r).lower()} reason={reason_r}")
        if valid_r:
            valid_candidates.append(("refiner", cleaned_ref))

    best_label, best_text = max(valid_candidates, key=lambda c: _score_candidate(c[1], src, tgt))
    print(f"[TRANSLATION_CHOSEN] balloon={balloon_tag} text=\"{best_text}\"")
    _store_cached(original, settings, src, tgt, best_text)
    return best_text


def translate_ja_to_en(text: str, config=None, balloon_id: int | None = None) -> str:
    """Backward-compat shim — prefer :func:`translate_text` directly."""
    return translate_text(text, source_lang="ja", target_lang="en", config=config, balloon_id=balloon_id)


# --- Validation --------------------------------------------------------------


def validate_translation_for_target(
    original: str,
    translated: str,
    source_lang: str,
    target_lang: str,
) -> tuple[bool, str]:
    """Return ``(is_valid, reason)`` for a candidate translation."""
    clean_original = safe_text(original).strip()
    clean_translated = safe_text(translated).strip()
    if not clean_translated or _is_punctuation_only(clean_translated):
        return False, "empty"
    if _looks_like_explanation(clean_translated):
        return False, "explanation_prefix"
    if _has_excessive_repetition(clean_translated):
        return False, "excessive_repetition"
    if clean_original and _is_too_long_for_balloon(clean_original, clean_translated):
        return False, "too_long_for_balloon"
    if _looks_like_ocr_noise(clean_translated):
        return False, "ocr_noise"
    if clean_original and clean_translated.casefold() == clean_original.casefold():
        # Identical to the source — accepted only when the original is
        # already in the target language (e.g. an English label inside a
        # Japanese page translated to English), or when source == target.
        if source_lang == target_lang:
            return True, "same_lang_passthrough"
        # en→pt special case: both are Latin script so the generic check is
        # ambiguous. Accept the passthrough only when the original clearly
        # looks like Portuguese already.
        if source_lang == "en" and target_lang == "pt":
            if _looks_like_portuguese(clean_original):
                return True, "same_as_original_already_in_target"
            return False, "same_as_original"
        if _text_is_in_language(clean_original, target_lang) and not _text_is_in_language(clean_original, source_lang):
            return True, "same_as_original_already_in_target"
        return False, "same_as_original"
    # The "still in source" check is only meaningful for non-Latin scripts.
    # For Latin-script source languages (en/pt/es/fr/...) we cannot tell apart
    # one from another with a script check — the empty/identical rules above
    # already cover the failure modes that matter.
    if source_lang in _NON_LATIN_LANGS and source_lang != target_lang and _text_is_in_language(clean_translated, source_lang):
        return False, f"still_in_source_lang:{source_lang}"
    # en→pt content-based check: reject candidates that come back clearly in
    # English (model echoed the source instead of translating).
    if source_lang == "en" and target_lang == "pt" and _looks_like_english(clean_translated) and not _looks_like_portuguese(clean_translated):
        return False, "still_in_source_lang:en"
    # Target-language mismatch: when target is a Latin language, make sure the
    # output isn't dominated by a different non-Latin script.
    if target_lang in {"en", "pt", "es", "fr", "de", "it", "nl", "pl", "tr", "vi"} and _has_foreign_script(clean_translated):
        return False, "wrong_target_lang"
    return True, "ok"


def validate_translation(original: str, translated: str, target_lang: str | None = None) -> tuple[bool, bool]:
    """Backward-compat wrapper used by :mod:`app.pipeline`.

    Returns ``(same, still_in_source)`` where ``still_in_source`` is True only
    if the candidate is detected to still be in the source language.
    """
    clean_original = safe_text(original)
    clean_translated = safe_text(translated).strip()
    same = bool(clean_original and clean_translated and clean_original.casefold() == clean_translated.casefold())
    src_guess = detect_language(clean_original, fallback="")
    if not src_guess:
        return same, False
    still_in_source = _text_is_in_language(clean_translated, src_guess) and src_guess != normalize_lang_code(target_lang or "", default="")
    return same, still_in_source


def contains_japanese(text: str) -> bool:
    """Legacy helper kept for compatibility — checks for any Japanese script."""
    return bool(_JAPANESE_RE.search(safe_text(text)) or _CJK_UNIFIED_RE.search(safe_text(text)))


def detect_language(text: str, *, fallback: str = "en") -> str:
    """Detect the dominant script in ``text`` and map it to a language code."""
    clean = safe_text(text)
    if not clean:
        return normalize_lang_code(fallback, default="en")
    if _JAPANESE_RE.search(clean):
        return "ja"
    if _HANGUL_RE.search(clean):
        return "ko"
    if _CYRILLIC_RE.search(clean):
        return "ru"
    if _ARABIC_RE.search(clean):
        return "ar"
    if _HEBREW_RE.search(clean):
        return "he"
    if _THAI_RE.search(clean):
        return "th"
    if _CJK_UNIFIED_RE.search(clean):
        # No kana — assume Chinese.
        return "zh"
    if _LATIN_LETTER_RE.search(clean):
        # Latin script — try a tiny content heuristic before defaulting.
        if _PT_DIACRITIC_RE.search(clean):
            return "pt"
        if _looks_like_portuguese(clean) and not _looks_like_english(clean):
            return "pt"
        if _looks_like_english(clean) and not _looks_like_portuguese(clean):
            return "en"
        return normalize_lang_code(fallback, default="en")
    return normalize_lang_code(fallback, default="en")


def _text_is_in_language(text: str, lang: str) -> bool:
    """Conservative per-language script check."""
    clean = safe_text(text)
    if not clean or not lang:
        return False
    lang = lang.lower()
    if lang == "ja":
        return bool(_JAPANESE_RE.search(clean) or _CJK_UNIFIED_RE.search(clean))
    if lang == "zh":
        return bool(_CJK_UNIFIED_RE.search(clean) and not _JAPANESE_RE.search(clean))
    if lang == "ko":
        return bool(_HANGUL_RE.search(clean))
    if lang == "ru":
        return bool(_CYRILLIC_RE.search(clean))
    if lang == "ar":
        return bool(_ARABIC_RE.search(clean))
    if lang == "he":
        return bool(_HEBREW_RE.search(clean))
    if lang == "th":
        return bool(_THAI_RE.search(clean))
    if lang == "pt":
        return _looks_like_portuguese(clean)
    if lang == "en":
        if not _LATIN_LETTER_RE.search(clean):
            return False
        if _JAPANESE_RE.search(clean) or _HANGUL_RE.search(clean) or _CYRILLIC_RE.search(clean) \
                or _ARABIC_RE.search(clean) or _HEBREW_RE.search(clean) or _THAI_RE.search(clean):
            return False
        # A string that clearly looks Portuguese is NOT English even though
        # both share the Latin alphabet.
        if _looks_like_portuguese(clean) and not _looks_like_english(clean):
            return False
        return True
    if lang in {"es", "fr", "de", "it", "nl", "pl", "tr", "vi"}:
        # For these the only reliable signal is "looks like latin" — we use it
        # only to detect that a translation is *not* still in a CJK source.
        return bool(_LATIN_LETTER_RE.search(clean)) and not (
            _JAPANESE_RE.search(clean) or _HANGUL_RE.search(clean) or _CYRILLIC_RE.search(clean)
            or _ARABIC_RE.search(clean) or _HEBREW_RE.search(clean) or _THAI_RE.search(clean)
        )
    return False


def _looks_like_portuguese(text: str) -> bool:
    """Content-based check: does this Latin string look like Portuguese?"""
    clean = safe_text(text)
    if not clean or not _LATIN_LETTER_RE.search(clean):
        return False
    if _JAPANESE_RE.search(clean) or _HANGUL_RE.search(clean) or _CYRILLIC_RE.search(clean) \
            or _ARABIC_RE.search(clean) or _HEBREW_RE.search(clean) or _THAI_RE.search(clean):
        return False
    if _PT_DIACRITIC_RE.search(clean):
        return True
    pt_hits = len(_PT_TOKEN_RE.findall(clean))
    en_hits = len(_EN_TOKEN_RE.findall(clean))
    if pt_hits == 0:
        return False
    # More pt tokens than en tokens, or pt tokens present in a very short
    # bubble where en tokens are absent.
    return pt_hits > en_hits or (pt_hits >= 1 and en_hits == 0)


def _looks_like_english(text: str) -> bool:
    clean = safe_text(text)
    if not clean or not _LATIN_LETTER_RE.search(clean):
        return False
    if _PT_DIACRITIC_RE.search(clean):
        return False
    en_hits = len(_EN_TOKEN_RE.findall(clean))
    pt_hits = len(_PT_TOKEN_RE.findall(clean))
    if en_hits == 0:
        return False
    return en_hits >= pt_hits


# --- Translation backends ----------------------------------------------------


def _translate_with(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    provider: str,
    model_id: str,
) -> str:
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider in {"multilingual", "local_hf", "hf", "transformers", "opus_mt", "small100"}:
        return _translate_transformers(text, source_lang, target_lang, model_id)
    if provider in {"ctranslate2", "ct2"}:
        return _translate_ctranslate2(text, source_lang, target_lang, model_id)
    # Unknown provider — try transformers as a last resort.
    return _translate_transformers(text, source_lang, target_lang, model_id)


def _translate_transformers(text: str, source_lang: str, target_lang: str, model_id: str) -> str:
    runtime = _get_transformers_runtime(model_id)
    tokenizer = runtime.tokenizer
    model = runtime.model
    torch = runtime.torch
    family = runtime.family

    num_beams = max(1, int(os.getenv("TRANSLATION_NUM_BEAMS", "1") or "1"))

    if family == "m2m100":
        try:
            tokenizer.src_lang = source_lang
        except Exception:
            pass
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoded = {k: v.to(runtime.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                max_new_tokens=192,
                num_beams=num_beams,
                early_stopping=(num_beams > 1),
            )
    elif family == "small100":
        try:
            tokenizer.tgt_lang = target_lang
        except Exception:
            pass
        try:
            tokenizer.src_lang = source_lang
        except Exception:
            pass
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoded = {k: v.to(runtime.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=192,
                num_beams=num_beams,
                early_stopping=(num_beams > 1),
            )
    else:
        # Marian / opus-mt / generic seq2seq — the model already encodes the
        # language pair, so we just translate the raw text.
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        encoded = {k: v.to(runtime.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=192,
                num_beams=num_beams,
                early_stopping=(num_beams > 1),
            )

    decoded = safe_text(tokenizer.decode(output_ids[0], skip_special_tokens=True)).strip()
    with _debug_lock:
        _last_debug["raw"] = safe_text(output_ids[0].detach().cpu().tolist())
        _last_debug["decoded"] = decoded
    return decoded


def _translate_ctranslate2(text: str, source_lang: str, target_lang: str, model_id: str) -> str:
    """SMALL100 / M2M100 served via CTranslate2 (optional dependency)."""
    try:
        import ctranslate2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional path
        raise TranslationError(
            "provider=ctranslate2 requer o pacote opcional 'ctranslate2'. "
            "Instale com: pip install ctranslate2"
        ) from exc

    runtime = _get_ct2_runtime(model_id)
    translator = runtime.model  # ctranslate2.Translator instance
    tokenizer = runtime.tokenizer

    try:
        tokenizer.src_lang = source_lang
    except Exception:
        pass
    try:
        tokenizer.tgt_lang = target_lang
    except Exception:
        pass

    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    target_prefix = None
    if hasattr(tokenizer, "lang_code_to_token"):
        token = tokenizer.lang_code_to_token.get(target_lang)
        if token is not None:
            target_prefix = [token]
    elif hasattr(tokenizer, "get_lang_id"):
        try:
            tok_id = tokenizer.get_lang_id(target_lang)
            target_prefix = [tokenizer.convert_ids_to_tokens(tok_id)]
        except Exception:
            target_prefix = None

    results = translator.translate_batch(
        [input_tokens],
        target_prefix=[target_prefix] if target_prefix else None,
        beam_size=4,
        max_decoding_length=192,
    )
    output_tokens = results[0].hypotheses[0]
    if target_prefix:
        output_tokens = output_tokens[len(target_prefix):]
    decoded = safe_text(tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)).strip()
    with _debug_lock:
        _last_debug["raw"] = safe_text(output_tokens)
        _last_debug["decoded"] = decoded
    return decoded


# --- Runtime loading ---------------------------------------------------------


def _detect_family(model_id: str) -> str:
    lower = model_id.lower()
    if "small100" in lower:
        return "small100"
    if "m2m100" in lower or "m2m_100" in lower:
        return "m2m100"
    return "marian"


def _get_transformers_runtime(model_id: str) -> _Seq2SeqRuntime:
    key = f"hf:{model_id}"
    with _runtime_lock:
        cached = _runtimes.get(key)
        if cached is not None:
            return cached

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device_name = "cuda" if torch.cuda.is_available() and os.getenv("TRANSLATION_DEVICE", "cpu").lower() != "cpu" else "cpu"
        device = torch.device(device_name)
        auth_kwargs = hf_auth_kwargs()
        load_kwargs = {"low_cpu_mem_usage": True}
        dtype_env = (os.getenv("TRANSLATION_DTYPE", "float32") or "float32").lower()
        if dtype_env in {"float16", "fp16", "half"}:
            load_kwargs["torch_dtype"] = torch.float16
        elif dtype_env in {"bfloat16", "bf16"}:
            load_kwargs["torch_dtype"] = torch.bfloat16

        family = _detect_family(model_id)

        try:
            if family == "m2m100":
                from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

                tokenizer = M2M100Tokenizer.from_pretrained(model_id, **auth_kwargs)
                model = M2M100ForConditionalGeneration.from_pretrained(model_id, **auth_kwargs, **load_kwargs)
            elif family == "small100":
                # SMALL100 ships its own tokenizer class; fall back to AutoTokenizer
                # with trust_remote_code if the dedicated class is unavailable.
                try:
                    from transformers import SMALL100Tokenizer  # type: ignore

                    tokenizer = SMALL100Tokenizer.from_pretrained(model_id, **auth_kwargs)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **auth_kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **auth_kwargs, **load_kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **auth_kwargs)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **auth_kwargs, **load_kwargs)
        except Exception as exc:
            if is_hf_auth_error(exc):
                raise TranslationError(hf_auth_help_message(model_id)) from exc
            raise

        model.to(device)
        model.eval()
        runtime = _Seq2SeqRuntime(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
            family=family,
            model_id=model_id,
        )
        _runtimes[key] = runtime
        print(f"[TRANSLATOR] loaded model={model_id} family={family} device={device_name}")
        return runtime


def _get_ct2_runtime(model_id: str) -> _Seq2SeqRuntime:
    key = f"ct2:{model_id}"
    with _runtime_lock:
        cached = _runtimes.get(key)
        if cached is not None:
            return cached

        import ctranslate2  # type: ignore
        from transformers import AutoTokenizer

        auth_kwargs = hf_auth_kwargs()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **auth_kwargs)
        except Exception as exc:
            if is_hf_auth_error(exc):
                raise TranslationError(hf_auth_help_message(model_id)) from exc
            raise

        device_name = "cuda" if os.getenv("TRANSLATION_DEVICE", "cpu").lower() != "cpu" else "cpu"
        try:
            translator = ctranslate2.Translator(model_id, device=device_name)
        except Exception as exc:
            raise TranslationError(f"falha ao carregar modelo CTranslate2 {model_id}: {exc}") from exc

        runtime = _Seq2SeqRuntime(
            tokenizer=tokenizer,
            model=translator,
            torch=None,
            device=device_name,
            family="ct2",
            model_id=model_id,
        )
        _runtimes[key] = runtime
        print(f"[TRANSLATOR] loaded model={model_id} family=ct2 device={device_name}")
        return runtime


# --- Cache helpers -----------------------------------------------------------


def _get_cached(text: str, settings: TranslationSettings, src: str, tgt: str) -> str | None:
    key = (settings.model, safe_text(text), src, tgt)
    with _translation_cache_lock:
        return _translation_cache.get(key)


def _store_cached(text: str, settings: TranslationSettings, src: str, tgt: str, translated: str) -> None:
    key = (settings.model, safe_text(text), src, tgt)
    with _translation_cache_lock:
        _translation_cache[key] = safe_text(translated)
        while len(_translation_cache) > _translation_cache_max_items:
            _translation_cache.pop(next(iter(_translation_cache)))


# --- Tiny utilities ----------------------------------------------------------


def _normalize_ocr_text(text: str) -> str:
    """Clean noisy OCR text before translation.

    Rules: join broken bubble lines, collapse repeated whitespace, drop
    stray isolated characters (single Latin letters/digits surrounded by
    spaces — common OCR junk on edges), keep punctuation, names and
    onomatopoeia intact.
    """
    cleaned = safe_text(text).replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    # Common OCR artefacts — full-width spaces, weird control chars.
    cleaned = cleaned.replace("　", " ").replace("﻿", "").replace("​", "")
    cleaned = re.sub(r"[•·∙]+", "", cleaned)  # bullet-like noise from OCR
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned
    # Strip stray isolated 1-char Latin tokens (e.g. "a", "I", "z") that
    # appear between CJK runs — they are almost always OCR garbage. Keep
    # uppercase "I" / "A" only if they sit between other Latin words.
    tokens = cleaned.split(" ")
    if len(tokens) > 1:
        kept: list[str] = []
        for idx, tok in enumerate(tokens):
            if len(tok) == 1 and tok.isascii() and tok.isalpha():
                prev_latin = idx > 0 and any(c.isalpha() and c.isascii() for c in tokens[idx - 1])
                next_latin = idx < len(tokens) - 1 and any(c.isalpha() and c.isascii() for c in tokens[idx + 1])
                if not (prev_latin or next_latin):
                    continue
            kept.append(tok)
        cleaned = " ".join(kept).strip()
    # Collapse runs of the same punctuation/dash/underscore artefacts.
    cleaned = re.sub(r"([_\-=])\1{2,}", r"\1", cleaned)
    return cleaned


_QUOTE_PAIRS = {
    "\"": "\"", "'": "'", "＂": "＂",
    "“": "”", "‘": "’",
    "「": "」", "『": "』",
    "(": ")", "[": "]", "{": "}",
}

_PREFIX_RE = re.compile(
    r"^(?:here(?:'s| is)?(?: the)?\s+(?:translation|translated text)\s*[:\-—]\s*"
    r"|translation\s*[:\-—]\s*"
    r"|translated\s*(?:text)?\s*[:\-—]\s*"
    r"|in\s+\w+\s+(?:it\s+)?(?:means|is)\s*[:\-—]\s*"
    r"|tradu[cç][aã]o\s*[:\-—]\s*"
    r"|aqui\s+est[aá]\s+a\s+tradu[cç][aã]o\s*[:\-—]\s*)",
    re.IGNORECASE,
)


def _clean_translation_output(text: str, target_lang: str) -> str:
    """Strip prefixes/quotes the model sometimes adds around the translation."""
    s = safe_text(text).strip()
    if not s:
        return s
    # Iterate: strip explanation prefixes and list markers, then unwrap quote
    # pairs (including asymmetric pairs like 「…」). Repeat in case wrappers
    # are nested.
    for _ in range(4):
        before = s
        s = _PREFIX_RE.sub("", s).strip()
        s = re.sub(r"^(?:[-*•]\s+|\d+[\.\)]\s+)", "", s).strip()
        if len(s) >= 2:
            opener = s[0]
            closer = _QUOTE_PAIRS.get(opener)
            if closer and s[-1] == closer:
                s = s[1:-1].strip()
        if s == before:
            break
    s = re.sub(r"\s+", " ", s).strip()
    return s


_EXPLANATION_PATTERNS = re.compile(
    r"(here(?:'s| is) (?:the )?translation|the translation is|in english(?: it)? (?:means|is)"
    r"|translates? to|this means|note:|explanation:|traduzido para|isto significa)",
    re.IGNORECASE,
)


def _looks_like_explanation(text: str) -> bool:
    return bool(_EXPLANATION_PATTERNS.search(text or ""))


def _has_excessive_repetition(text: str) -> bool:
    s = safe_text(text)
    if len(s) < 8:
        return False
    # Same char repeated 8+ times (mojibake / decoding loops).
    if re.search(r"(.)\1{7,}", s):
        return True
    # Same 2-6 char chunk repeated 4+ times back-to-back ("lalalalala").
    if re.search(r"(.{2,6}?)\1{3,}", s):
        return True
    # Same word repeated 4+ times consecutively.
    if re.search(r"\b(\w{2,})(?:\s+\1\b){3,}", s, flags=re.IGNORECASE):
        return True
    return False


def _is_too_long_for_balloon(original: str, translated: str) -> bool:
    # Absolute cap — manga lines are short. Allow more headroom for CJK
    # source (one CJK char ≈ a word).
    if len(translated) > 400:
        return True
    src_units = max(1, len(original))
    # Translation more than 8× the source length is almost always the model
    # hallucinating or emitting an explanation paragraph.
    return len(translated) > src_units * 8 and len(translated) > 80


def _looks_like_ocr_noise(text: str) -> bool:
    s = safe_text(text)
    if len(s) < 4:
        return False
    alpha = sum(1 for c in s if c.isalpha())
    return alpha == 0


_FOREIGN_SCRIPT_RE = re.compile(
    r"[぀-ゟ゠-ヿ㐀-䶿一-鿿가-힯ᄀ-ᇿЀ-ӿ؀-ۿ֐-׿฀-๿]"
)


def _has_foreign_script(text: str) -> bool:
    """Latin-target output contains a meaningful run of non-Latin script."""
    hits = _FOREIGN_SCRIPT_RE.findall(text or "")
    return len(hits) >= 2


def _score_candidate(text: str, source_lang: str, target_lang: str) -> float:
    """Higher is better — used to pick among valid candidates."""
    s = safe_text(text)
    if not s:
        return -1e6
    score = 0.0
    # Correct script for the target.
    if target_lang in _NON_LATIN_LANGS:
        if _text_is_in_language(s, target_lang):
            score += 5.0
    else:
        if _LATIN_LETTER_RE.search(s):
            score += 5.0
        if _has_foreign_script(s):
            score -= 8.0
    # Avoid still-in-source.
    if source_lang in _NON_LATIN_LANGS and _text_is_in_language(s, source_lang):
        score -= 10.0
    # Length sweet spot for manga bubbles (2–120 chars).
    length = len(s)
    if 2 <= length <= 120:
        score += 2.0
    elif length > 200:
        score -= 2.0
    # Repetition penalty.
    if _has_excessive_repetition(s):
        score -= 5.0
    # Explanation penalty.
    if _looks_like_explanation(s):
        score -= 5.0
    return score


# --- Optional LLM refiner ----------------------------------------------------

_REFINER_PROMPT = (
    "Improve this manga dialogue translation so it sounds natural and coherent "
    "in the target language. Preserve the original meaning, tone, emotion, "
    "names, punctuation, and short speech-bubble style. Do not add explanations. "
    "Return only the final translated line."
)


def _refiner_settings() -> dict | None:
    provider = safe_text(os.getenv("TRANSLATION_REFINER_PROVIDER")).lower()
    if not provider or provider in {"none", "off", "disabled"}:
        return None
    api_key = safe_text(os.getenv("TRANSLATION_REFINER_API_KEY"))
    if not api_key:
        return None
    if provider in {"openai", "openai_compat", "openai-compatible"}:
        return {
            "provider": "openai",
            "api_key": api_key,
            "model": safe_text(os.getenv("TRANSLATION_REFINER_MODEL")) or "gpt-4o-mini",
            "base_url": safe_text(os.getenv("TRANSLATION_REFINER_BASE_URL")) or "https://api.openai.com/v1",
        }
    if provider in {"anthropic", "claude"}:
        return {
            "provider": "anthropic",
            "api_key": api_key,
            "model": safe_text(os.getenv("TRANSLATION_REFINER_MODEL")) or "claude-haiku-4-5-20251001",
            "base_url": safe_text(os.getenv("TRANSLATION_REFINER_BASE_URL")) or "https://api.anthropic.com/v1",
        }
    return None


def _refine_with_llm(
    *,
    original: str,
    initial: str,
    source_lang: str,
    target_lang: str,
    context: list[str],
    balloon_id: int,
) -> str | None:
    cfg = _refiner_settings()
    if cfg is None:
        return None
    print(f"[TRANSLATION_REFINER] balloon={balloon_id} provider={cfg['provider']}")
    user_payload_lines = [
        f"source_lang: {source_lang}",
        f"target_lang: {target_lang}",
        f"original_ocr: {original}",
        f"initial_translation: {initial}",
    ]
    if context:
        joined_ctx = " | ".join(context[:6])
        user_payload_lines.append(f"page_context: {joined_ctx}")
    user_payload = "\n".join(user_payload_lines)
    try:
        import requests  # type: ignore

        timeout = float(os.getenv("TRANSLATION_REFINER_TIMEOUT", "20") or "20")
        if cfg["provider"] == "openai":
            resp = requests.post(
                f"{cfg['base_url'].rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "temperature": 0.3,
                    "max_tokens": 256,
                    "messages": [
                        {"role": "system", "content": _REFINER_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return safe_text(data["choices"][0]["message"]["content"])
        if cfg["provider"] == "anthropic":
            resp = requests.post(
                f"{cfg['base_url'].rstrip('/')}/messages",
                headers={
                    "x-api-key": cfg["api_key"],
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "max_tokens": 256,
                    "system": _REFINER_PROMPT,
                    "messages": [{"role": "user", "content": user_payload}],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            blocks = data.get("content") or []
            for block in blocks:
                if block.get("type") == "text":
                    return safe_text(block.get("text"))
            return None
    except Exception as exc:
        print(f"[TRANSLATION_QUALITY_ERROR] balloon={balloon_id} error=refiner:{exc}")
        return None
    return None


def _is_punctuation_only(text: str) -> bool:
    clean = safe_text(text).strip()
    if not clean:
        return False
    return all(ch in string.punctuation for ch in clean)
