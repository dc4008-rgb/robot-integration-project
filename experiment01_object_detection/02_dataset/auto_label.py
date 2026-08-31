"""Automatically pre-label images in raw/ with a COCO-pretrained model.

Prerequisite: The class names in data.yaml must match COCO class names
(cup / mouse / keyboard / bottle are all included in COCO). The script maps
COCO class IDs to the IDs in data.yaml and writes YOLO-format .txt files.

Usage:
    python auto_label.py                    # Pre-label all unlabeled images
    python auto_label.py --conf 0.4 --overwrite
    python auto_label.py --input-dir recollection/train

Afterward, open raw/ in labelImg and review every image: add missed objects,
delete false detections, and correct boxes. Pre-labeling reduces manual work
but does not replace manual verification.
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically pre-label images with a COCO-pretrained model"
    )
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        help="A larger model is more accurate for pre-labeling; m is better than n",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument(
        "--input-dir", type=Path, default=RAW,
        help="Directory containing images to pre-label",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .txt files"
    )
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
        print(
            f"Warning: {sorted(missing)} are not COCO classes; "
            "these classes must be labeled entirely by hand"
        )
    if not coco_to_mine:
        raise SystemExit(
            "None of the configured classes can be pre-labeled with the COCO model"
        )
    print(
        "Classes available for pre-labeling: "
        f"{sorted(n for n in my_names if n in model.names.values())}"
    )

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(
            f"No images found in {input_dir}; run capture_images.py in the "
            "data-collection directory first"
        )

    todo = [p for p in images if args.overwrite or not p.with_suffix(".txt").exists()]
    print(f"Found {len(images)} images; {len(todo)} awaiting pre-labeling\n")

    n_boxes = 0
    empty = []
    for i, img in enumerate(todo, 1):
        result = model.predict(str(img), conf=args.conf, verbose=False)[0]
        lines = []
        for box in result.boxes:
            coco_id = int(box.cls[0])
            if coco_id not in coco_to_mine:
                continue
            # xywhn contains normalized center coordinates, width, and height.
            x, y, w, h = (float(v) for v in box.xywhn[0])
            lines.append(f"{coco_to_mine[coco_id]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        img.with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
        if not lines:
            empty.append(img.name)
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")

    print(
        f"\nDone: {n_boxes} boxes, "
        f"{n_boxes / max(len(todo), 1):.1f} per image on average"
    )
    if empty:
        print(
            f"WARNING: No target was detected in {len(empty)} images; "
            "prioritize them for manual labeling:"
        )
        for name in empty[:10]:
            print(f"    {name}")
    print(
        f"\nNext: Open {input_dir} in a labeling tool, review and correct every "
        "image, then run split_dataset.py"
    )


if __name__ == "__main__":
    main()
