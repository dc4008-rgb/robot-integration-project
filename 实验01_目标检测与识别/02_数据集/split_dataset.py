"""把标注完成的图片+标签划分成 train/val，整理成 ultralytics 要求的目录结构。

标注完成后的输入布局（labelImg / X-AnyLabeling 导出 YOLO 格式）:
    02_数据集/raw/xxx.jpg
    02_数据集/raw/xxx.txt

输出:
    02_数据集/images/train, images/val, labels/train, labels/val

用法:
    python split_dataset.py --val-ratio 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="划分 YOLO 数据集")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    images = sorted(p for p in RAW.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"{RAW} 里没有图片")

    paired, unlabeled = [], []
    for img in images:
        label = img.with_suffix(".txt")
        (paired if label.exists() else unlabeled).append(img)

    if unlabeled:
        print(f"警告: {len(unlabeled)} 张图没有对应 .txt 标签，已跳过")
        for p in unlabeled[:5]:
            print(f"  - {p.name}")

    if not paired:
        raise SystemExit("没有任何已标注的图片")

    random.Random(args.seed).shuffle(paired)
    n_val = max(1, int(len(paired) * args.val_ratio))
    splits = {"val": paired[:n_val], "train": paired[n_val:]}

    for split, items in splits.items():
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        for d in (img_dir, lbl_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        for img in items:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(img.with_suffix(".txt"), lbl_dir / f"{img.stem}.txt")
        print(f"{split}: {len(items)} 张")

    print(f"完成，总计 {len(paired)} 张已标注图片")


if __name__ == "__main__":
    main()
