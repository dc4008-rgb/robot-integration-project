"""把 best.pt 导出成 Jetson 上的 TensorRT 引擎，推理速度通常能提升 2-4 倍。

必须在 Jetson 本机上运行（TensorRT 引擎不能跨设备复用）。

用法:
    python export_engine.py --weights best.pt --half
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 TensorRT 引擎")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16，Orin 上建议开")
    args = parser.parse_args()

    model = YOLO(args.weights)
    path = model.export(format="engine", imgsz=args.imgsz, half=args.half, device=0)
    print(f"导出完成: {path}")
    print("在 yolo_ros2_node.py 里把 --weights 换成这个 .engine 文件即可")


if __name__ == "__main__":
    main()
