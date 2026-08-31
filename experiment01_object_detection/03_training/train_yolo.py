"""Train a YOLO model on the server (AutoDL).

Usage (on the server):
    python train_yolo.py --data /root/robot_det/02_dataset/data.yaml --epochs 150

Training output: runs/detect/<name>/weights/best.pt
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 object detection model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights; n/s/m are progressively larger, more accurate, and slower")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="desktop_det")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-period", type=int, default=-1)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        # These augmentations cover lighting and angle variations common to desktop objects.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0,
    )

    metrics = model.val()
    print("\n===== Validation Metrics =====")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    for i, name in enumerate(cfg["names"].values()):
        print(f"  {name:10s} AP50={metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
