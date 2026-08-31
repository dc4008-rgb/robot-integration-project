"""Export best.pt to a TensorRT engine on Jetson, typically improving inference speed by 2-4x.

This must run on the Jetson itself because TensorRT engines are not portable across devices.

Usage:
    python export_engine.py --weights best.pt --half
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a TensorRT engine")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Enable FP16; recommended on Orin")
    args = parser.parse_args()

    model = YOLO(args.weights)
    path = model.export(format="engine", imgsz=args.imgsz, half=args.half, device=0)
    print(f"Export complete: {path}")
    print("Set --weights in yolo_ros2_node.py to this .engine file")


if __name__ == "__main__":
    main()
