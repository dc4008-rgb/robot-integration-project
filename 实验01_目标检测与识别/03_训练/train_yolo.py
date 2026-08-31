"""在服务器（AutoDL）上训练 YOLO 模型。

用法（服务器上）:
    python train_yolo.py --data /root/robot_det/02_数据集/data.yaml --epochs 150

训练产物: runs/detect/<name>/weights/best.pt
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 目标检测模型")
    parser.add_argument("--data", required=True, help="data.yaml 路径")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练权重，n/s/m 依次更大更准更慢")
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
        # 桌面物体常见的光照/角度变化，靠这几项增强覆盖
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0,
    )

    metrics = model.val()
    print("\n===== 验证集指标 =====")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    for i, name in enumerate(cfg["names"].values()):
        print(f"  {name:10s} AP50={metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
