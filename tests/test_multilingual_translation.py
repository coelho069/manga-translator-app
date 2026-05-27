from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig
from app.translator import TransformersSeq2SeqTranslator, validate_translation_output


def main() -> None:
    translator = TransformersSeq2SeqTranslator(
        AppConfig(
            translation_provider="m2m100",
            translation_model="facebook/m2m100_418M",
            source_lang="auto",
            target_lang="pt",
            translation_cache_enabled=False,
        )
    )

    cases = [
        ("ja", "en", "こんにちは"),
        ("ja", "en", "ありがとう"),
        ("en", "pt", "I will protect you."),
        ("en", "pt", "Don't underestimate me!"),
        ("es", "pt", "No me subestimes."),
        ("zh", "en", "你好"),
    ]

    for source_lang, target_lang, text in cases:
        print(f"[TRANSLATION_INPUT] text=\"{text}\"")
        translated = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
        print(f"[TRANSLATION_OUTPUT] text=\"{translated}\"")
        valid, reason = validate_translation_output(text, translated, source_lang, target_lang)
        print(f"[TRANSLATION_VALIDATE] valid={valid} reason={reason}")
        if not valid:
            raise AssertionError(f"{source_lang}->{target_lang} invalid: {reason}: {translated}")


if __name__ == "__main__":
    main()
