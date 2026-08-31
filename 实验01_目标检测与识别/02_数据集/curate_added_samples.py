#!/usr/bin/env python3
"""整理“增添样本”中的耳机困难样本和保温瓶正样本。"""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["cup", "mouse", "keyboard", "bottle"]

SAMPLES = {
    "微信图片_20260831102231_1275_7.jpg": {
        "sha256": "8402cce7d95f50f16b07f99e7383da306fb7a1b89978194e235291a6b7a04eb8",
        "output": "actual_headphones_case_1275.jpg",
        "usage": "train_pure_negative",
        "labels": [],
    },
    "微信图片_20260831102235_1276_7.jpg": {
        "sha256": "b594eefee6c3673105b4dda842b5017206216d312908be1724895d4c4cadf66f",
        "usage": "excluded",
        "reason": "Top-edge object may be a cup or bottle; unsafe as an empty label.",
    },
    "微信图片_20260831102237_1277_7.jpg": {
        "sha256": "48bbcd9f2c8017107cb5ed3e18e6ffd62a0a1b90bbed6399176afd0d7414a4fc",
        "usage": "excluded",
        "reason": "Ambiguous cup/bottle and mouse-like objects at image edges.",
    },
    "微信图片_20260831102239_1278_7.jpg": {
        "sha256": "c5607c59d564dd9b6ef260e12516d70b5509661bffab013cdcddced7a944d819",
        "output": "actual_headphones_case_1278.jpg",
        "usage": "train_mixed",
        "labels": [
            [2, 0.818354, 0.026619, 0.173413, 0.053132],
            [2, 0.935027, 0.251718, 0.129023, 0.234997],
        ],
    },
    "微信图片_20260831102242_1279_7.jpg": {
        "sha256": "66103a7803be7031661d910e3a9fd045ec3c347cfbeb5cb95e83bdde5208d0fc",
        "output": "actual_headphones_case_1279.jpg",
        "usage": "train_pure_negative",
        "labels": [],
    },
    "微信图片_20260831102243_1280_7.jpg": {
        "sha256": "c4798f0e9fe63efc0f084de871933e4d856a4a2e02413d1ba1056733c12e8029",
        "output": "actual_headphones_case_1280.jpg",
        "usage": "train_pure_negative",
        "labels": [],
    },
    "微信图片_20260831102246_1281_7.jpg": {
        "sha256": "b0f4efa01a259fa7ee55dcf6d76785a86470307f5c818fa225c021290356d326",
        "output": "actual_headphones_case_1281.jpg",
        "usage": "train_pure_negative",
        "labels": [],
    },
    "微信图片_20260831102248_1282_7.jpg": {
        "sha256": "e6e9cf3cfa270e5d6b11ea29568f77c3409c758f6ab7599bccdd834e7beca4dd",
        "output": "actual_headphones_case_1282.jpg",
        "usage": "regression_mixed",
        "labels": [
            [2, 0.910000, 0.030000, 0.130000, 0.050000],
            [2, 0.980000, 0.210000, 0.040000, 0.050000],
        ],
    },
    "微信图片_20260831102734_1283_7.jpg": {
        "sha256": "6cd0ec5ae8f8f811ac855f493c518469bb6f7d7a1754db59426f85ad97a73ebf",
        "output": "actual_thermos_1283.jpg",
        "usage": "train_positive",
        "labels": [
            [3, 0.202293, 0.508670, 0.353176, 0.728668],
            [3, 0.756364, 0.519752, 0.450519, 0.794177],
            [2, 0.217236, 0.575276, 0.433004, 0.183063],
            [2, 0.458099, 0.482327, 0.190425, 0.240669],
            [2, 0.936479, 0.498063, 0.125827, 0.144698],
        ],
    },
    "微信图片_20260831102738_1284_7.jpg": {
        "sha256": "da2b8c65154670c9aafeaf9abcde4b32f73ce06c42f7875e0fe64451172d0f35",
        "output": "actual_thermos_1284.jpg",
        "usage": "train_positive",
        "labels": [
            [3, 0.384998, 0.356054, 0.298313, 0.439020],
            [3, 0.655379, 0.467975, 0.277612, 0.404018],
        ],
    },
    "微信图片_20260831102741_1285_7.jpg": {
        "sha256": "df0275542182f4d2c9b8abfa193f05f9cdde6a4b6ad87193da83f0a67acb9f01",
        "output": "actual_thermos_1285.jpg",
        "usage": "train_positive",
        "labels": [
            [3, 0.532003, 0.424581, 0.387683, 0.301686],
            [3, 0.545264, 0.649258, 0.384391, 0.210832],
        ],
    },
    "微信图片_20260831102744_1286_7.jpg": {
        "sha256": "7fdd21c60522490b45eb2f68a6541a38584375742fab067b67a4e63f3e06d51c",
        "output": "actual_thermos_1286.jpg",
        "usage": "train_positive",
        "labels": [
            [3, 0.469156, 0.607178, 0.319188, 0.436748],
            [3, 0.621945, 0.487532, 0.227100, 0.385162],
        ],
    },
    "微信图片_20260831102746_1287_7.jpg": {
        "sha256": "f7b3afc97f7ba7d1fb96be949095c2bdf52fe85ebd823b82d834b4646142d533",
        "output": "actual_thermos_1287.jpg",
        "usage": "train_positive",
        "labels": [
            [3, 0.490000, 0.410000, 0.520000, 0.280000],
            [3, 0.450000, 0.690000, 0.610000, 0.220000],
        ],
    },
}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Curate user-captured headphones and thermos samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root.parents[1] / "增添样本",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "recollection" / "added_samples_20260831",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_text(labels):
    lines = []
    for class_id, center_x, center_y, width, height in labels:
        if class_id not in range(len(CLASS_NAMES)):
            raise SystemExit(f"Invalid class id: {class_id}")
        values = (center_x, center_y, width, height)
        if not all(0 <= value <= 1 for value in values):
            raise SystemExit(f"Invalid normalized label values: {values}")
        if width <= 0 or height <= 0:
            raise SystemExit(f"Non-positive label size: {values}")
        if center_x - width / 2 < 0 or center_x + width / 2 > 1:
            raise SystemExit(f"Label exceeds horizontal image bounds: {values}")
        if center_y - height / 2 < 0 or center_y + height / 2 > 1:
            raise SystemExit(f"Label exceeds vertical image bounds: {values}")
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def prepare_output(output: Path, overwrite: bool):
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    for usage in sorted({sample["usage"] for sample in SAMPLES.values()} - {"excluded"}):
        (output / usage).mkdir(parents=True)


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    prepare_output(output, args.overwrite)

    records = []
    for source_name, sample in SAMPLES.items():
        source_image = source / source_name
        if not source_image.is_file():
            raise SystemExit(f"Missing source image: {source_image}")
        actual_hash = sha256_file(source_image)
        if actual_hash != sample["sha256"]:
            raise SystemExit(
                f"Source hash mismatch for {source_name}: "
                f"{actual_hash} != {sample['sha256']}"
            )

        record = {"source_filename": source_name, **sample}
        if sample["usage"] != "excluded":
            destination = output / sample["usage"] / sample["output"]
            shutil.copy2(source_image, destination)
            destination.with_suffix(".txt").write_text(
                label_text(sample["labels"]), encoding="utf-8"
            )
            if sha256_file(destination) != actual_hash:
                raise SystemExit(f"Copied image hash mismatch: {destination}")
            record["output_image_sha256"] = actual_hash
        records.append(record)

    usage_counts = Counter(record["usage"] for record in records)
    box_counts = Counter(
        CLASS_NAMES[label[0]]
        for sample in SAMPLES.values()
        for label in sample.get("labels", [])
        if sample["usage"].startswith("train_")
    )
    manifest = {
        "selection_version": 1,
        "source_directory": str(source),
        "class_names": CLASS_NAMES,
        "visual_audit": {
            "headphones_cases": (
                "Empty labels are used only when no project target is visible; "
                "mixed scenes retain keyboard labels."
            ),
            "thermos_bottles": (
                "Both gray and blue thermos bottles are labeled as bottle."
            ),
            "coordinate_system": (
                "Normalized coordinates follow the EXIF-oriented image shown by "
                "OpenCV and Ultralytics."
            ),
        },
        "counts": {
            "by_usage": dict(sorted(usage_counts.items())),
            "training_boxes": {
                class_name: box_counts[class_name] for class_name in CLASS_NAMES
            },
        },
        "records": records,
    }
    (output / "selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()