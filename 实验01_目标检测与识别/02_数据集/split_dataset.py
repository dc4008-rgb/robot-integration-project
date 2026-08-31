"""把标注完成的图片+标签划分成 train/val，整理成 ultralytics 要求的目录结构。

标注完成后的输入布局（labelImg / X-AnyLabeling 导出 YOLO 格式）:
    02_数据集/raw/xxx.jpg
    02_数据集/raw/xxx.txt

输出:
    02_数据集/images/train, images/val, labels/train, labels/val

用法:
    python split_dataset.py --val-ratio 0.2
    python split_dataset.py --raw-dir ../训练数据集 --val-ratio 0.2
    python split_dataset.py --raw-dir ../训练数据集 --external-yolo-dir external_openimages_yolo
    python split_dataset.py --raw-dir ../训练数据集 --negative-dir recollection/negative_train
"""
import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_CLASSES = ["cup", "mouse", "keyboard", "bottle"]


def collect_paired(root: Path, description: str):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{description}目录不存在: {root}")

    images = sorted(path for path in root.iterdir() if path.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"{root} 里没有图片")

    paired = [image for image in images if image.with_suffix(".txt").exists()]
    unlabeled = [image for image in images if not image.with_suffix(".txt").exists()]
    if unlabeled:
        print(f"警告: {description}中 {len(unlabeled)} 张图没有对应 .txt 标签，已跳过")
        for path in unlabeled[:5]:
            print(f"  - {path.name}")
    if not paired:
        raise SystemExit(f"{description}中没有任何已标注图片")
    return paired


def collect_negatives(root: Path, description: str):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{description}目录不存在: {root}")

    images = sorted(path for path in root.iterdir() if path.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"{root} 里没有图片")

    labeled = [
        image for image in images
        if image.with_suffix(".txt").exists()
        and image.with_suffix(".txt").read_text().strip()
    ]
    if labeled:
        examples = ", ".join(path.name for path in labeled[:5])
        raise SystemExit(
            f"{description}只能包含不含目标的纯负样本，发现非空标签: {examples}"
        )
    return images


def reject_exact_overlap(train, val):
    train_hashes = {
        hashlib.sha256(image.read_bytes()).digest(): image
        for image in train
    }
    duplicates = [
        (train_hashes[digest], image)
        for image in val
        if (digest := hashlib.sha256(image.read_bytes()).digest()) in train_hashes
    ]
    if duplicates:
        examples = ", ".join(f"{train.name} / {val.name}" for train, val in duplicates[:3])
        raise SystemExit(f"训练集与独立验证集存在相同图片: {examples}")


def reject_name_collisions(images):
    seen = {}
    collisions = []
    for image in images:
        if image.stem in seen:
            collisions.append((seen[image.stem], image))
        else:
            seen[image.stem] = image
    if collisions:
        examples = ", ".join(f"{first} / {second}" for first, second in collisions[:3])
        raise SystemExit(f"训练来源中存在同名图片，无法安全合并: {examples}")


def materialize(source: Path, destination: Path, hardlink: bool) -> None:
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def validate_external_roots(roots):
    validated = []
    for root in roots:
        root = root.expanduser().resolve()
        images = root / "images" / "train"
        labels = root / "labels" / "train"
        yaml_path = root / "dataset.yaml"
        if not images.is_dir() or not labels.is_dir() or not yaml_path.is_file():
            raise SystemExit(
                f"外部数据集缺少 images/train、labels/train 或 dataset.yaml: {root}"
            )

        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = config.get("names") if isinstance(config, dict) else None
        if isinstance(names, dict):
            names = [value for _, value in sorted((int(key), value) for key, value in names.items())]
        if names != EXPECTED_CLASSES:
            raise SystemExit(
                f"外部数据集类别顺序不匹配: {names}，应为 {EXPECTED_CLASSES}"
            )
        validated.append((root, images, labels))
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description="划分 YOLO 数据集")
    parser.add_argument("--raw-dir", type=Path, default=RAW, help="图片和标签所在目录")
    parser.add_argument(
        "--train-extra-dir", type=Path, action="append", default=[],
        help="追加的本地训练图片与标签目录，可重复指定",
    )
    parser.add_argument(
        "--negative-dir", type=Path, action="append", default=[],
        help="不含四类目标的纯负样本图片目录，可重复指定；仅加入训练集并生成空标签",
    )
    parser.add_argument(
        "--val-dir", type=Path,
        help="独立拍摄并标注的验证目录；指定后 raw-dir 中的图片全部用于训练",
    )
    parser.add_argument(
        "--external-yolo-dir",
        type=Path,
        action="append",
        default=[],
        help="额外 YOLO 数据集根目录，可重复指定；只加入训练集",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-hardlink", action="store_true",
        help="训练集使用同卷硬链接以节省本地空间；验证集仍复制",
    )
    args = parser.parse_args()

    paired = collect_paired(args.raw_dir, "主训练来源")
    for index, train_dir in enumerate(args.train_extra_dir, start=1):
        paired.extend(collect_paired(train_dir, f"追加训练来源 {index}"))
    negatives = []
    for index, negative_dir in enumerate(args.negative_dir, start=1):
        negatives.extend(collect_negatives(negative_dir, f"纯负样本来源 {index}"))
    reject_name_collisions(paired + negatives)

    external_roots = validate_external_roots(args.external_yolo_dir)

    if args.val_dir:
        val = collect_paired(args.val_dir, "独立验证集")
        reject_exact_overlap(paired + negatives, val)
        splits = {"val": val, "train": list(paired)}
    else:
        rng = random.Random(args.seed)
        rng.shuffle(paired)
        n_val = max(1, int(len(paired) * args.val_ratio))

        classes = {
            image: {
                int(line.split()[0])
                for line in image.with_suffix(".txt").read_text().splitlines()
                if line.strip()
            }
            for image in paired
        }
        uncovered = set().union(*classes.values())
        val = []
        while uncovered and len(val) < n_val:
            candidates = [image for image in paired if image not in val]
            best = max(candidates, key=lambda image: len(classes[image] & uncovered))
            if not classes[best] & uncovered:
                break
            val.append(best)
            uncovered -= classes[best]
        remaining = [image for image in paired if image not in val]
        val.extend(remaining[: n_val - len(val)])
        if uncovered:
            print(f"警告: 验证集容量不足，未覆盖类别 {sorted(uncovered)}")
        splits = {"val": val, "train": [image for image in paired if image not in val]}

    splits["train"].extend(negatives)
    negative_set = set(negatives)

    output_dirs = {}
    for split in splits:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        for d in (img_dir, lbl_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        output_dirs[split] = (img_dir, lbl_dir)

    for split, items in splits.items():
        img_dir, lbl_dir = output_dirs[split]
        for img in items:
            use_hardlink = split == "train" and args.train_hardlink
            materialize(img, img_dir / img.name, use_hardlink)
            destination_label = lbl_dir / f"{img.stem}.txt"
            if img in negative_set:
                destination_label.write_text("")
            else:
                materialize(img.with_suffix(".txt"), destination_label, use_hardlink)

    external_count = 0
    train_img_dir, train_lbl_dir = output_dirs["train"]
    for source_index, (_, external_images, external_labels) in enumerate(
        external_roots, start=1
    ):
        for image in sorted(
            path for path in external_images.iterdir() if path.suffix.lower() in IMG_EXTS
        ):
            label = external_labels / f"{image.stem}.txt"
            if not label.exists():
                print(f"警告: 外部图片 {image.name} 没有标签，已跳过")
                continue
            output_stem = f"external{source_index}_{image.stem}"
            materialize(
                image,
                train_img_dir / f"{output_stem}{image.suffix.lower()}",
                args.train_hardlink,
            )
            materialize(
                label, train_lbl_dir / f"{output_stem}.txt", args.train_hardlink
            )
            external_count += 1

    print(f"val: {len(splits['val'])} 张（仅本地数据）")
    print(
        f"train: {len(splits['train']) + external_count} 张"
        f"（本地有目标 {len(splits['train']) - len(negatives)}"
        f" + 纯负样本 {len(negatives)} + 外部 {external_count}）"
    )

    print(
        f"完成，本地有目标 {len(paired)} 张，纯负样本 {len(negatives)} 张，"
        f"外部 {external_count} 张"
    )


if __name__ == "__main__":
    main()
