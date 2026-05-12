from __future__ import annotations

from functools import lru_cache

from app.config import AppConfig
from app.utils import safe_text


class BaseTranslator:
    def translate(self, text: str) -> str:
        raise NotImplementedError


class IdentityTranslator(BaseTranslator):
    def translate(self, text: str) -> str:
        return safe_text(text)


class GoogleTextTranslator(BaseTranslator):
    def __init__(self, config: AppConfig):
        self.config = config

    def translate(self, text: str) -> str:
        clean = safe_text(text)
        if not clean:
            return ""
        return self._translate_cached(clean, self.config.source_lang, self.config.target_lang)

    @staticmethod
    @lru_cache(maxsize=512)
    def _translate_cached(text: str, source_lang: str, target_lang: str) -> str:
        try:
            from deep_translator import GoogleTranslator

            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return safe_text(translated) or text
        except Exception:
            return text

