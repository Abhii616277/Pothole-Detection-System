"""
Train a YOLOv8 pothole detector.

Dataset: 665 annotated road images (pothole vs. no pothole), YOLO format,
originally curated by Atikur Rahman Chitholian, split 70/20/10 train/val/test.
See dataset/data.yaml for paths and class names.

Usage:
    python train.py --epochs 60 --imgsz 640 --batch 16
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for first GPU, 'cpu' for CPU")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--name", default="pothole_yolov8n")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project="runs",
        name=args.name,
    )

    # Copy the best checkpoint into models/ so app/main.py can find it.
    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    if best.exists():
        (out_dir / "best.pt").write_bytes(best.read_bytes())
        print(f"Saved best checkpoint to {out_dir / 'best.pt'}")

    metrics = model.val(data=args.data)
    print("Validation metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
