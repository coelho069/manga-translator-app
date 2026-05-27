from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.manga_ocr_engine import configure_manga_ocr, run_manga_ocr


def console_safe(value: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def find_example_image() -> Path | None:
    examples_dir = ROOT / "examples"
    if not examples_dir.exists():
        return None
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in examples_dir.glob(pattern):
            return path
    return None


def make_smoke_image() -> Image.Image:
    image = Image.new("RGB", (360, 120), "white")
    draw = ImageDraw.Draw(image)
    font = None
    for font_path in (
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ):
        try:
            font = ImageFont.truetype(font_path, 48)
            break
        except Exception:
            continue
    draw.text((28, 30), "\u30c6\u30b9\u30c8", fill="black", font=font)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Testa o modelo kha-white/manga-ocr-base.")
    parser.add_argument("image", nargs="?", help="Caminho opcional para imagem de teste.")
    args = parser.parse_args()

    configure_manga_ocr(
        model_name="kha-white/manga-ocr-base",
        cache_dir=str(ROOT / "models" / "huggingface"),
        device="cpu",
    )

    try:
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")
            image = Image.open(image_path).convert("RGB")
            print(f"Imagem: {image_path}")
        else:
            image_path = find_example_image()
            if image_path is not None:
                image = Image.open(image_path).convert("RGB")
                print(f"Imagem: {image_path}")
            else:
                image = make_smoke_image()
                print("Imagem: smoke test gerado em memoria")

        text = run_manga_ocr(image)
        print(console_safe(f"Texto reconhecido: {text or ''}"))
        return 0 if text is not None else 2
    except Exception as exc:
        print(console_safe(f"Erro ao carregar/testar Manga OCR: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
