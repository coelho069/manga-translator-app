from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera predicoes visuais de debug para um modelo YOLO segmentation.")
    parser.add_argument("--model", default="models/bubble_seg.pt", help="Caminho do modelo .pt.")
    parser.add_argument("--source", required=True, help="Imagem ou pasta de imagens.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianca minima.")
    parser.add_argument("--output", default="output/debug_predictions", help="Pasta de saida.")
    return parser.parse_args()


def list_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Fonte nao encontrada: {source}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    images = list_sources(source_path)
    if not images:
        raise ValueError(f"Nenhuma imagem encontrada em: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    for image_path in images:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            device="cpu",
            verbose=False,
        )
        if not results:
            continue
        plotted = results[0].plot()
        output_path = output_dir / f"{image_path.stem}_debug.png"

        import cv2

        ok = cv2.imwrite(str(output_path), plotted)
        if not ok:
            raise ValueError(f"Nao foi possivel salvar: {output_path}")
        print(f"Salvo: {output_path}")

    print(f"Predicoes de debug salvas em: {output_dir}")


if __name__ == "__main__":
    main()

