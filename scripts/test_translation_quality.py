"""Smoke-test the translation-quality pipeline on 5 manga balloons.

Runs three pipelines per sample so the developer can compare:
  1. raw OCR text;
  2. initial machine translation (the configured TRANSLATION_PROVIDER);
  3. final refined output (after validation + optional LLM refiner).

Pass --offline to skip the heavy seq2seq model and instead use a tiny stub
translator (useful on machines without GPU or model weights). The stub still
exercises validation, refiner, cleanup and logging — the parts changed in
this task.

Usage:
  python scripts/test_translation_quality.py
  python scripts/test_translation_quality.py --offline
  TRANSLATION_REFINER_PROVIDER=openai TRANSLATION_REFINER_API_KEY=... \
      python scripts/test_translation_quality.py --offline
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import translator as T  # noqa: E402


SAMPLES = [
    # (source_lang, target_lang, ocr_text, page_context)
    ("ja", "en", "こんにちは、元気ですか？", ["お久しぶり！", "今日は良い天気ですね"]),
    ("ja", "pt", "おはよう！\n  今日は学校に行く。", ["眠い…", "もう朝？"]),
    ("ja", "en", "や... やめてくれ！", ["うわああ！", "助けて！"]),
    ("ja", "en", "強くなりたい・・・！", ["俺は", "もっと"]),
    ("ja", "en", "ありがとう、 ・ 友よ", ["これで…", "終わりだ"]),
]


def _install_stub_translator() -> None:
    """Replace the heavy seq2seq backend with a deterministic stub."""

    fake_outputs = {
        "こんにちは、元気ですか？": "hello, how are you?",
        "おはよう！ 今日は学校に行く。": "Bom dia! Hoje vou para a escola.",
        "や... やめてくれ！": "S-stop it!",
        "強くなりたい・・・！": "I want to get stronger...!",
        "ありがとう、 友よ": "Thanks, my friend",
    }

    def fake_translate_with(*, text, source_lang, target_lang, provider, model_id):
        # Simulate noisy MT output for at least one balloon so validation/
        # cleanup actually fire.
        if text in fake_outputs:
            return fake_outputs[text]
        return f"Here is the translation: {text}"

    T._translate_with = fake_translate_with  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="use stub translator (no model load)")
    args = parser.parse_args()

    if args.offline:
        _install_stub_translator()
        os.environ.setdefault("TRANSLATION_PROVIDER", "stub")

    print("\n=== TRANSLATION QUALITY TEST ===\n")
    failures = 0
    for idx, (src, tgt, ocr, ctx) in enumerate(SAMPLES, start=1):
        print(f"--- balloon {idx} ({src}->{tgt}) ---")
        print(f"OCR original   : {ocr!r}")
        cleaned = T._normalize_ocr_text(ocr)
        print(f"OCR limpo      : {cleaned!r}")
        try:
            initial = T._translate_with(
                text=cleaned,
                source_lang=src,
                target_lang=tgt,
                provider=os.getenv("TRANSLATION_PROVIDER", "multilingual"),
                model_id=os.getenv("TRANSLATION_MODEL", "facebook/m2m100_418M"),
            )
        except Exception as exc:
            initial = f"<ERRO: {exc}>"
        print(f"Traducao bruta : {initial!r}")
        try:
            final = T.translate_text(ocr, source_lang=src, target_lang=tgt, balloon_id=idx, context=ctx)
        except Exception as exc:
            final = f"<ERRO: {exc}>"
            failures += 1
        print(f"Traducao final : {final!r}\n")

    print(f"falhas={failures}/{len(SAMPLES)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
