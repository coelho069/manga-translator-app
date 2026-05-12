from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina/fine-tuna um modelo YOLO segmentation para baloes de fala.")
    parser.add_argument("--model", default="models/bubble_seg.pt", help="Modelo base .pt para iniciar o treino.")
    parser.add_argument("--data", default="dataset/bubbles/data.yaml", help="Arquivo data.yaml do dataset.")
    parser.add_argument("--epochs", type=int, default=50, help="Numero de epocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem de treino.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Dispositivo: cpu, 0, 0,1 etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo base nao encontrado: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de dataset nao encontrado: {data_path}")

    if str(args.device).lower() == "cpu":
        print("Aviso: treino em CPU funciona, mas pode ser muito lento. Use GPU se estiver disponivel.")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/segment",
        name="train",
        task="segment",
    )

    print("Treino concluido. Procure o melhor peso em runs/segment/train/weights/best.pt")


if __name__ == "__main__":
    main()

