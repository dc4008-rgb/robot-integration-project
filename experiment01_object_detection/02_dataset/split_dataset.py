"""Split labeled image-label pairs into train/val sets for Ultralytics.

Input layout after labeling (YOLO format exported by labelImg / X-AnyLabeling):
    <dataset-dir>/raw/xxx.jpg
    <dataset-dir>/raw/xxx.txt

Output:
    <dataset-dir>/images/train, images/val, labels/train, labels/val

Usage:
    python split_dataset.py --val-ratio 0.2
    python split_dataset.py --raw-dir ../path/to/training_dataset --val-ratio 0.2
    python split_dataset.py --raw-dir ../path/to/training_dataset --external-yolo-dir external_openimages_yolo
    python split_dataset.py --raw-dir ../path/to/training_dataset --negative-dir recollection/negative_train
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
        raise SystemExit(f"{description} directory does not exist: {root}")

    images = sorted(path for path in root.iterdir() if path.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {root}")

    paired = [image for image in images if image.with_suffix(".txt").exists()]
    unlabeled = [image for image in images if not image.with_suffix(".txt").exists()]
    if unlabeled:
        print(
            f"Warning: {len(unlabeled)} images in {description} have no matching "
            ".txt label and were skipped"
        )
        for path in unlabeled[:5]:
            print(f"  - {path.name}")
    if not paired:
        raise SystemExit(f"No labeled images found in {description}")
    return paired


def collect_negatives(root: Path, description: str):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{description} directory does not exist: {root}")

    images = sorted(path for path in root.iterdir() if path.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {root}")

    labeled = [
        image for image in images
        if image.with_suffix(".txt").exists()
        and image.with_suffix(".txt").read_text().strip()
    ]
    if labeled:
        examples = ", ".join(path.name for path in labeled[:5])
        raise SystemExit(
            f"{description} must contain only pure negative samples with no target "
            f"objects; found non-empty labels: {examples}"
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
        raise SystemExit(
            f"Identical images found in the training and independent validation "
            f"sets: {examples}"
        )


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
        raise SystemExit(
            f"Images with duplicate basenames found across training sources; "
            f"cannot merge safely: {examples}"
        )


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
                f"External dataset is missing images/train, labels/train, or "
                f"dataset.yaml: {root}"
            )

        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        names = config.get("names") if isinstance(config, dict) else None
        if isinstance(names, dict):
            names = [value for _, value in sorted((int(key), value) for key, value in names.items())]
        if names != EXPECTED_CLASSES:
            raise SystemExit(
                f"External dataset class order does not match: {names}; "
                f"expected {EXPECTED_CLASSES}"
            )
        validated.append((root, images, labels))
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a YOLO dataset")
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW,
        help="Directory containing images and labels",
    )
    parser.add_argument(
        "--train-extra-dir", type=Path, action="append", default=[],
        help="Additional local training image and label directory; may be repeated",
    )
    parser.add_argument(
        "--negative-dir", type=Path, action="append", default=[],
        help=(
            "Pure-negative image directory containing none of the four target "
            "classes; may be repeated; added only to training with empty labels"
        ),
    )
    parser.add_argument(
        "--val-dir", type=Path,
        help=(
            "Independently captured and labeled validation directory; when set, "
            "all images in raw-dir are used for training"
        ),
    )
    parser.add_argument(
        "--external-yolo-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional YOLO dataset root; may be repeated; added only to training",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-hardlink", action="store_true",
        help=(
            "Use same-volume hard links for the training set to save local space; "
            "the validation set is still copied"
        ),
    )
    args = parser.parse_args()

    paired = collect_paired(args.raw_dir, "primary training source")
    for index, train_dir in enumerate(args.train_extra_dir, start=1):
        paired.extend(collect_paired(train_dir, f"additional training source {index}"))
    negatives = []
    for index, negative_dir in enumerate(args.negative_dir, start=1):
        negatives.extend(
            collect_negatives(negative_dir, f"pure-negative source {index}")
        )
    reject_name_collisions(paired + negatives)

    external_roots = validate_external_roots(args.external_yolo_dir)

    if args.val_dir:
        val = collect_paired(args.val_dir, "independent validation set")
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
            print(
                f"Warning: The validation set is too small to cover classes "
                f"{sorted(uncovered)}"
            )
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
                print(
                    f"Warning: External image {image.name} has no label and was skipped"
                )
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

    print(f"val: {len(splits['val'])} images (local data only)")
    print(
        f"train: {len(splits['train']) + external_count} images"
        f" (local images with targets {len(splits['train']) - len(negatives)}"
        f" + pure negatives {len(negatives)} + external {external_count})"
    )

    print(
        f"Done: {len(paired)} local images with targets, "
        f"{len(negatives)} pure negatives, and {external_count} external images"
    )


if __name__ == "__main__":
    main()
