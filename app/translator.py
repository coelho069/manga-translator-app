from __future__ import annotations

import re
import threading
from functools import lru_cache

from app.config import AppConfig
from app.glossary import GLOSSARY
from app.utils import resolve_translation_lang, safe_text


IDIOMATIC_MAP = {
    "no way": "nao pode ser",
    "i got it": "entendi",
    "got it": "entendi",
    "shut up": "cala a boca",
    "damn it": "droga",
    "what the hell": "que diabos",
    "are you okay": "voce esta bem",
    "leave it to me": "deixa comigo",
    "i'm counting on you": "conto com voce",
    "i will protect you": "eu vou te proteger",
    "don't give up": "nao desista",
}

JA_HONORIFICS = [
    "さん",
    "くん",
    "ちゃん",
    "先輩",
    "先生",
    "様",
]

COMMON_INITIAL_WORDS = {
    "A",
    "An",
    "And",
    "Are",
    "But",
    "Come",
    "Did",
    "Do",
    "Does",
    "Don",
    "Go",
    "He",
    "Hey",
    "His",
    "How",
    "I",
    "If",
    "It",
    "Let",
    "No",
    "Oh",
    "Please",
    "She",
    "So",
    "That",
    "The",
    "Then",
    "They",
    "This",
    "Wait",
    "We",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "You",
}

TITLE_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Lord|Lady|Princess|Prince|King|Queen|Sir|Master)\.\s+[A-Z][a-zA-Z'-]*"
    r"|\b(?:Mr|Mrs|Ms|Dr|Lord|Lady|Princess|Prince|King|Queen|Sir|Master)\s+[A-Z][a-zA-Z'-]*"
)
CAPITALIZED_RE = re.compile(r"\b[A-Z][a-zA-Z'-]{2,}\b")


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


class GoogleTextTranslator(BaseTranslator):
    _client_cache = {}
    _client_lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.config = config
        self.translation_mode = safe_text(config.translation_mode) or "en_to_pt"
        self.source_lang = resolve_translation_lang(config.source_lang)
        self.target_lang = resolve_translation_lang(config.target_lang)
        self.translation_style = self._normalize_style(config.translation_style)
        self._page_memory: dict[str, str] = {}

    def translate(self, text: str) -> str:
        clean = safe_text(text)
        if not clean:
            return ""
        if clean in self._page_memory:
            return self._page_memory[clean]
        translated = self._translate_one(clean)
        self._page_memory[clean] = translated
        return translated

    def translate_batch(self, texts: list[str]) -> list[str]:
        clean_texts = [safe_text(text) for text in texts]
        unique_to_translate: list[str] = []
        seen = set()
        for text in clean_texts:
            if not text or text in self._page_memory or text in seen:
                continue
            seen.add(text)
            unique_to_translate.append(text)

        for text in unique_to_translate:
            translated = self._translate_one(text)
            self._page_memory[text] = translated

        return [self._page_memory.get(text, "") if text else "" for text in clean_texts]

    def _translate_one(self, text: str) -> str:
        original = safe_text(text)
        if not original:
            return ""

        idiomatic = self._idiomatic_translation(original)
        if idiomatic:
            return idiomatic

        protected_text, placeholders = self._protect_terms(original)
        try:
            translated = self._translate_cached(protected_text, self.source_lang, self.target_lang)
        except Exception:
            translated = protected_text

        translated = self._restore_terms(translated, placeholders)
        translated = normalize_translation_text(translated, self.target_lang)
        translated = preserve_terminal_punctuation(original, translated, self.target_lang)
        return translated or original

    def _idiomatic_translation(self, text: str) -> str:
        if self.translation_style != "natural" or self.source_lang != "en" or self.target_lang != "pt":
            return ""

        body, punctuation = _split_terminal_punctuation(text)
        key = safe_text(body).lower()
        mapped = IDIOMATIC_MAP.get(key)
        if not mapped:
            return ""
        return normalize_translation_text(mapped + punctuation, "pt")

    def _protect_terms(self, text: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        protected = text

        if self.source_lang == "en":
            protected, placeholders = _protect_matches(protected, GLOSSARY.keys(), placeholders)
            names = detect_proper_names(protected)
            protected, placeholders = _protect_matches(protected, names, placeholders)
            return protected, placeholders

        if self.source_lang == "ja":
            protected, placeholders = _protect_matches(protected, JA_HONORIFICS, placeholders)
            return protected, placeholders

        return protected, placeholders

    @staticmethod
    def _restore_terms(text: str, placeholders: dict[str, str]) -> str:
        restored = safe_text(text)
        for placeholder, value in placeholders.items():
            restored = restored.replace(placeholder, value)
            restored = restored.replace(placeholder.lower(), value)
            restored = restored.replace(placeholder.replace("_", " "), value)
        return restored

    @staticmethod
    def _normalize_style(value) -> str:
        style = safe_text(value).lower()
        if style == "literal":
            return "literal"
        return "natural"

    @staticmethod
    @lru_cache(maxsize=2048)
    def _translate_cached(text: str, source_lang: str, target_lang: str) -> str:
        try:
            translator = GoogleTextTranslator._get_client(source_lang, target_lang)
            translated = translator.translate(text)
            return safe_text(translated) or text
        except Exception:
            return text

    @classmethod
    def _get_client(cls, source_lang: str, target_lang: str):
        key = (source_lang, target_lang)
        with cls._client_lock:
            if key not in cls._client_cache:
                from deep_translator import GoogleTranslator

                cls._client_cache[key] = GoogleTranslator(source=source_lang, target=target_lang)
            return cls._client_cache[key]


def detect_proper_names(text: str) -> list[str]:
    clean = safe_text(text)
    if not clean:
        return []

    names: list[str] = []
    for match in TITLE_NAME_RE.finditer(clean):
        names.append(match.group(0))

    for match in CAPITALIZED_RE.finditer(clean):
        word = match.group(0)
        if match.start() == 0 and word in COMMON_INITIAL_WORDS:
            continue
        if word in COMMON_INITIAL_WORDS:
            continue
        if word.upper() == word:
            continue
        names.append(word)

    return _unique_by_length(names)


def normalize_translation_text(text: str, target_lang: str) -> str:
    normalized_target = resolve_translation_lang(target_lang)
    if normalized_target == "pt":
        return normalize_translation_pt(text)
    return normalize_translation_en(text)


def normalize_translation_pt(text: str) -> str:
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


def normalize_translation_en(text: str) -> str:
    clean = safe_text(text)
    if not clean:
        return ""

    clean = clean.replace("…", "...")
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    clean = re.sub(r"\s*\.\s*\.\s*\.", "...", clean)
    clean = re.sub(r"\.{4,}", "...", clean)
    clean = re.sub(r"!{3,}", "!!", clean)
    clean = re.sub(r"\?{3,}", "??", clean)
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
    return normalize_translation_text(clean_translated + source_punctuation, target_lang)


def _protect_matches(text: str, values, placeholders: dict[str, str]) -> tuple[str, dict[str, str]]:
    protected = text
    for value in _unique_by_length([safe_text(item) for item in values if safe_text(item)]):
        if value not in protected:
            continue
        placeholder = f"__NAME_{len(placeholders)}__"
        protected = protected.replace(value, placeholder)
        placeholders[placeholder] = value
    return protected, placeholders


def _unique_by_length(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in sorted(values, key=len, reverse=True):
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _split_terminal_punctuation(text: str) -> tuple[str, str]:
    clean = safe_text(text)
    if not clean:
        return "", ""

    for raw, mapped in ("！？", "?!"), ("？！", "?!"), ("。", "."), ("！", "!"), ("？", "?"), ("…", "..."):
        if clean.endswith(raw):
            return clean[: -len(raw)], mapped

    match = re.search(r"([!?.]+|\?!|!\?)$", clean)
    if not match:
        return clean, ""
    return clean[: match.start()], match.group(0)
