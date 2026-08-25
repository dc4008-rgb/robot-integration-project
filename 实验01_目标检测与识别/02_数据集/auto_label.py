"""用 COCO 预训练模型自动预标注 raw/ 里的图片，人工只需检查修正。

前提：data.yaml 里的类别名与 COCO 类别名一致（cup / mouse / keyboard / bottle 都在 COCO 里）。
脚本自动把 COCO 类别 id 映射成 data.yaml 里的 id，输出 YOLO 格式 .txt。

用法:
    python auto_label.py                    # 预标注全部未标注图片
    python auto_label.py --conf 0.4 --overwrite

之后务必用 labelImg 打开 raw/ 逐张检查：补漏检、删误检、修框。
预标注只是省力，不能代替人工核对。
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO 预训练模型自动预标注")
    parser.add_argument("--model", default="yolov8m.pt", help="预标注用大模型更准，m 比 n 好")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .txt")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
    my_names = {name: idx for idx, name in cfg["names"].items()}

    model = YOLO(args.model)
    coco_to_mine = {
        coco_id: my_names[name]
        for coco_id, name in model.names.items()
        if name in my_names
    }
    missing = set(my_names) - set(model.names.values())
    if missing:
        print(f"注意: {sorted(missing)} 不在 COCO 类别里，这些类必须全手工标注")
    if not coco_to_mine:
        raise SystemExit("没有任何类别能用 COCO 模型预标注")
    print(f"可预标注类别: {sorted(n for n in my_names if n in model.names.values())}")

    images = sorted(p for p in RAW.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"{RAW} 里没有图片，先跑 01_数据采集/capture_images.py")

    todo = [p for p in images if args.overwrite or not p.with_suffix(".txt").exists()]
    print(f"共 {len(images)} 张图，待预标注 {len(todo)} 张\n")

    n_boxes = 0
    empty = []
    for i, img in enumerate(todo, 1):
        result = model.predict(str(img), conf=args.conf, verbose=False)[0]
        lines = []
        for box in result.boxes:
            coco_id = int(box.cls[0])
            if coco_id not in coco_to_mine:
                continue
            # xywhn 已经是归一化的中心点+宽高，正是 YOLO 标签格式
            x, y, w, h = (float(v) for v in box.xywhn[0])
            lines.append(f"{coco_to_mine[coco_id]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        img.with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
        if not lines:
            empty.append(img.name)
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")

    print(f"\n完成：{n_boxes} 个框，平均每张 {n_boxes / max(len(todo), 1):.1f} 个")
    if empty:
        print(f"⚠️  {len(empty)} 张没检出任何目标，需要重点手工标注：")
        for name in empty[:10]:
            print(f"    {name}")
    print("\n下一步：labelImg raw/  逐张检查修正，然后跑 split_dataset.py")


if __name__ == "__main__":
    main()
