#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path


TARGET_CLASSES = ["cup", "mouse", "keyboard", "bottle"]
SOURCE_CLASSES = [
    ("Coffee cup", "cup", "cup"),
    ("Computer mouse", "mouse", "mouse"),
    ("Computer keyboard", "keyboard", "keyboard"),
    ("Bottle", "bottle", "bottle"),
]


def parse_args():
    default_output = Path(__file__).resolve().parent / "external_openimages_yolo"
    parser = argparse.ArgumentParser(
        description="Download a class-filtered Open Images V7 subset and export YOLO labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cup", type=int, default=20, help="Cup images to request")
    parser.add_argument("--mouse", type=int, default=20, help="Mouse images to request")
    parser.add_argument("--keyboard", type=int, default=20, help="Keyboard images to request")
    parser.add_argument("--bottle", type=int, default=50, help="Bottle images to request")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    parser.add_argument("--output", type=Path, default=default_output, help="YOLO export directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export")
    return parser.parse_args()


def validate_args(args):
    quotas = {name: getattr(args, name) for _, _, name in SOURCE_CLASSES}
    if any(value < 0 for value in quotas.values()):
        raise SystemExit("Sample counts must be non-negative.")
    if not any(quotas.values()):
        raise SystemExit("At least one sample count must be greater than zero.")
    return quotas


def import_fiftyone(cache_dir):
    os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(cache_dir / "database"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    try:
        import setuptools  # noqa: F401 - restores distutils for ETA on Python 3.13
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise SystemExit(
            "FiftyOne is required. Install it with: "
            "python -m pip install fiftyone setuptools"
        ) from exc
    return fo, foz


def prepare_output(output, overwrite):
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output is not empty: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def download_and_merge(fo, foz, quotas, seed, cache_dir):
    fo.config.dataset_zoo_dir = str(cache_dir)
    source_to_target = {source: target for source, target, _ in SOURCE_CLASSES}
    combined_name = f"experiment1-openimages-combined-{seed}"
    if fo.dataset_exists(combined_name):
        fo.delete_dataset(combined_name)
    combined = fo.Dataset(combined_name)
    temporary_names = []

    try:
        for source_class, _, quota_name in SOURCE_CLASSES:
            quota = quotas[quota_name]
            if quota == 0:
                continue

            dataset_name = f"experiment1-openimages-{quota_name}-{seed}-{quota}"
            temporary_names.append(dataset_name)
            subset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],
                classes=[source_class],
                only_matching=False,
                max_samples=quota,
                shuffle=True,
                seed=seed,
                dataset_name=dataset_name,
                drop_existing_dataset=True,
            )
            filtered = subset.filter_labels(
                "ground_truth",
                fo.ViewField("label").is_in(list(source_to_target)),
            )
            mapped = filtered.map_labels("ground_truth", source_to_target)
            combined.merge_samples(mapped.select_fields("ground_truth"))

        for sample in combined.iter_samples(autosave=True):
            unique = {}
            for detection in sample.ground_truth.detections:
                key = (detection.label, tuple(detection.bounding_box))
                unique.setdefault(key, detection)
            sample.ground_truth.detections = list(unique.values())

        counts = combined.count_values("ground_truth.detections.label")
        unexpected = sorted(set(counts) - set(TARGET_CLASSES))
        if unexpected:
            raise RuntimeError(f"Unexpected labels after mapping: {unexpected}")
        if not counts:
            raise RuntimeError("No detection labels were downloaded.")
        return combined, temporary_names, counts
    except Exception:
        combined.delete()
        for name in temporary_names:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
        raise


def export_yolo(fo, dataset, output, counts, quotas, seed):
    dataset.export(
        export_dir=str(output),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train",
        classes=TARGET_CLASSES,
    )
    source_info = {
        "dataset": "Open Images V7",
        "source": "https://storage.googleapis.com/openimages/web/index.html",
        "annotations_license": "CC BY 4.0",
        "images_license": "Listed as CC BY 2.0; verify individual image metadata before redistribution",
        "source_to_target": {source: target for source, target, _ in SOURCE_CLASSES},
        "target_classes": TARGET_CLASSES,
        "requested_images_per_class": quotas,
        "exported_box_counts": counts,
        "seed": seed,
    }
    (output / "source.json").write_text(
        json.dumps(source_info, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    quotas = validate_args(args)
    output = prepare_output(args.output, args.overwrite)
    cache_dir = output.parent / ".fiftyone_openimages_cache"
    fo, foz = import_fiftyone(cache_dir)
    combined = None
    temporary_names = []

    try:
        combined, temporary_names, counts = download_and_merge(
            fo, foz, quotas, args.seed, cache_dir
        )
        export_yolo(fo, combined, output, counts, quotas, args.seed)
        print(f"Exported {len(combined)} images to {output}")
        print("Box counts:", ", ".join(f"{name}={counts.get(name, 0)}" for name in TARGET_CLASSES))
    finally:
        if combined is not None:
            combined.delete()
        for name in temporary_names:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)


if __name__ == "__main__":
    main()