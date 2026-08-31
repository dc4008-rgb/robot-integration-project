#!/usr/bin/env python3
"""下载并初筛 Open Images 正样本与相似干扰物候选图。

脚本直接使用 Open Images 可视化器的按类别索引，避免下载全量元数据。
正样本导出四类 YOLO 标签；负样本仅保留没有四类目标正标签的图片并生成空标签。
Open Images 标注不保证穷尽，候选集仍需模型挖掘和人工复核后才能用于训练。
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path


INDEX_BASE = (
    "https://storage.googleapis.com/openimages/web_v6/visualizer/"
    "annotations_detection_train"
)
IMAGE_BASE = "https://open-images-dataset.s3.amazonaws.com/train"
TARGET_CLASSES = {
    "Coffee cup": ("/m/02p5f1q", 0, "cup"),
    "Computer mouse": ("/m/020lf", 1, "mouse"),
    "Computer keyboard": ("/m/01m2v", 2, "keyboard"),
    "Bottle": ("/m/04dr76w", 3, "bottle"),
}
DISTRACTOR_CLASSES = {
    "Mobile phone": ("/m/050k8", "mobile_phone", "mobile_phone"),
    "Headphones": ("/m/01b7fy", "headphones", "headphones"),
    "Power plugs and sockets": ("/m/03bbps", "power_plugs", "power_plugs"),
}
IMAGE_ID_PATTERN = re.compile(r"([0-9a-f]{16})$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download bidirectional Open Images candidates without FiftyOne.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--positive-per-class", type=int, default=20)
    parser.add_argument("--mobile-phone", type=int, default=40)
    parser.add_argument("--headphones", type=int, default=30)
    parser.add_argument("--power-plugs", type=int, default=30)
    parser.add_argument("--min-positive-box-area", type=float, default=0.01)
    parser.add_argument("--min-distractor-box-area", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--output", type=Path,
        default=root / "external_openimages_bidirectional_candidates",
    )
    parser.add_argument(
        "--exclude-dir", type=Path, action="append", default=[],
        help="包含既有图片的目录，按 Open Images ID 和 SHA-256 去重，可重复指定",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
    counts = [
        args.positive_per_class,
        args.mobile_phone,
        args.headphones,
        args.power_plugs,
    ]
    if any(count < 0 for count in counts):
        raise SystemExit("Sample counts must be non-negative.")
    if not any(counts):
        raise SystemExit("At least one sample count must be greater than zero.")
    for name in ("min_positive_box_area", "min_distractor_box_area"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 1].")


def prepare_output(output: Path, overwrite: bool):
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output is not empty: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    (output / "positive" / "images" / "train").mkdir(parents=True)
    (output / "positive" / "labels" / "train").mkdir(parents=True)
    (output / "negative_train").mkdir(parents=True)
    return output


def request_bytes(url: str, timeout: int = 45):
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "3", "--connect-timeout", "15", "--max-time", str(timeout),
        "--user-agent", "robot-integration-course/1.0", url,
    ]
    try:
        return subprocess.run(
            command, check=True, capture_output=True, timeout=timeout + 10
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"curl failed for {url}: {detail.strip()}") from exc


def fetch_class_index(mid: str):
    filename = mid.replace("/", "_") + ".json"
    data = request_bytes(f"{INDEX_BASE}/{filename}")
    return json.loads(data)


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_images_in(paths):
    ids = set()
    hashes = set()
    for root in paths:
        root = root.expanduser().resolve()
        if not root.exists():
            raise SystemExit(f"Exclude directory does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                hashes.add(hash_file(path))
                match = IMAGE_ID_PATTERN.search(path.stem)
                if match:
                    ids.add(match.group(1))
    return ids, hashes


def positive_image_labels(record):
    return {
        label["category"]
        for label in record.get("image_labels", [])
        if float(label.get("confidence", 0)) == 1.0
    }


def target_objects(record):
    return [
        obj for obj in record.get("objects", [])
        if obj.get("text") in TARGET_CLASSES
    ]


def has_requested_object(record, class_name):
    return any(obj.get("text") == class_name for obj in record.get("objects", []))


def box_area(obj):
    box = obj["bounding_box"]
    return max(0.0, float(box["xmax"]) - float(box["xmin"])) * max(
        0.0, float(box["ymax"]) - float(box["ymin"])
    )


def yolo_lines(objects):
    lines = []
    for obj in objects:
        box = obj["bounding_box"]
        xmin = float(box["xmin"])
        xmax = float(box["xmax"])
        ymin = float(box["ymin"])
        ymax = float(box["ymax"])
        if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
            raise ValueError(f"Invalid bounding box: {box}")
        class_id = TARGET_CLASSES[obj["text"]][1]
        lines.append(
            f"{class_id} {(xmin + xmax) / 2:.6f} {(ymin + ymax) / 2:.6f} "
            f"{xmax - xmin:.6f} {ymax - ymin:.6f}"
        )
    return lines


def license_metadata(record):
    image = record["image"]
    footer = image.get("footnote_bottom_right", "")
    urls = re.findall(r"href=['\"]([^'\"]+)", footer)
    return {
        "thumbnail_url": image.get("url"),
        "full_resolution_url": image.get("url_full_res"),
        "license_url": next((url for url in urls if "creativecommons.org" in url), None),
        "attribution_html": footer,
        "title_html": image.get("footnote_top_right", ""),
    }


def download_image(image_id: str, record, destination: Path):
    image = record["image"]
    errors = []
    for url in (
        f"{IMAGE_BASE}/{image_id}.jpg",
        image.get("url"),
        image.get("url_full_res"),
    ):
        if not url:
            continue
        try:
            data = request_bytes(url)
            if len(data) < 1024 or data[:2] != b"\xff\xd8":
                raise ValueError(f"Unexpected image response: {len(data)} bytes")
            destination.write_bytes(data)
            return url, hashlib.sha256(data).hexdigest()
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def shuffled_entries(index, seed, class_name):
    entries = list(index.items())
    random.Random(f"{seed}:{class_name}").shuffle(entries)
    return entries


def select_positives(output, excluded_ids, excluded_hashes, args):
    selected = []
    rejected = []
    selected_ids = set(excluded_ids)
    selected_hashes = set(excluded_hashes)
    image_dir = output / "positive" / "images" / "train"
    label_dir = output / "positive" / "labels" / "train"

    for class_name, (mid, _, slug) in TARGET_CLASSES.items():
        if args.positive_per_class == 0:
            continue
        count = 0
        index = fetch_class_index(mid)
        for image_id, record in shuffled_entries(index, args.seed, class_name):
            if count >= args.positive_per_class:
                break
            if image_id in selected_ids:
                rejected.append({"image_id": image_id, "reason": "existing_or_selected"})
                continue
            objects = target_objects(record)
            boxed_classes = {obj["text"] for obj in objects}
            missing_boxes = (positive_image_labels(record) & TARGET_CLASSES.keys()) - boxed_classes
            if missing_boxes:
                rejected.append({
                    "image_id": image_id,
                    "reason": "positive_target_without_box",
                    "classes": sorted(missing_boxes),
                })
                continue
            if not has_requested_object(record, class_name):
                continue
            requested = [obj for obj in objects if obj["text"] == class_name]
            if max(map(box_area, requested), default=0.0) < args.min_positive_box_area:
                rejected.append({"image_id": image_id, "reason": "requested_object_too_small"})
                continue
            if float(record["image"].get("rotation", 0)) != 0:
                rejected.append({"image_id": image_id, "reason": "nonzero_rotation"})
                continue
            try:
                labels = yolo_lines(objects)
                filename = f"{slug}_{image_id}.jpg"
                used_url, digest = download_image(
                    image_id, record, image_dir / filename
                )
            except Exception as exc:
                rejected.append({"image_id": image_id, "reason": "download_or_label_error", "detail": str(exc)})
                continue
            if digest in selected_hashes:
                (image_dir / filename).unlink()
                rejected.append({
                    "image_id": image_id,
                    "reason": "exact_duplicate_sha256",
                    "sha256": digest,
                })
                continue
            (label_dir / f"{Path(filename).stem}.txt").write_text(
                "\n".join(labels) + "\n", encoding="utf-8"
            )
            selected_ids.add(image_id)
            selected_hashes.add(digest)
            count += 1
            selected.append({
                "kind": "positive",
                "requested_class": class_name,
                "filename": filename,
                "image_id": image_id,
                "target_box_counts": {
                    target: sum(obj["text"] == target for obj in objects)
                    for target in TARGET_CLASSES
                },
                "downloaded_url": used_url,
                "sha256": digest,
                **license_metadata(record),
            })
    return selected, rejected, selected_ids, selected_hashes


def select_negatives(output, selected_ids, selected_hashes, args):
    quotas = {
        "mobile_phone": args.mobile_phone,
        "headphones": args.headphones,
        "power_plugs": args.power_plugs,
    }
    selected = []
    rejected = []
    image_dir = output / "negative_train"

    for class_name, (mid, slug, quota_name) in DISTRACTOR_CLASSES.items():
        if quotas[quota_name] == 0:
            continue
        count = 0
        index = fetch_class_index(mid)
        for image_id, record in shuffled_entries(index, args.seed, class_name):
            if count >= quotas[quota_name]:
                break
            if image_id in selected_ids:
                rejected.append({"image_id": image_id, "reason": "existing_or_selected"})
                continue
            target_labels = positive_image_labels(record) & TARGET_CLASSES.keys()
            target_boxes = target_objects(record)
            if target_labels or target_boxes:
                rejected.append({
                    "image_id": image_id,
                    "reason": "known_project_target",
                    "classes": sorted(target_labels | {obj["text"] for obj in target_boxes}),
                })
                continue
            if not has_requested_object(record, class_name):
                continue
            requested = [
                obj for obj in record.get("objects", []) if obj.get("text") == class_name
            ]
            if max(map(box_area, requested), default=0.0) < args.min_distractor_box_area:
                rejected.append({"image_id": image_id, "reason": "distractor_too_small"})
                continue
            if float(record["image"].get("rotation", 0)) != 0:
                rejected.append({"image_id": image_id, "reason": "nonzero_rotation"})
                continue
            try:
                filename = f"{slug}_{image_id}.jpg"
                used_url, digest = download_image(
                    image_id, record, image_dir / filename
                )
            except Exception as exc:
                rejected.append({"image_id": image_id, "reason": "download_error", "detail": str(exc)})
                continue
            if digest in selected_hashes:
                (image_dir / filename).unlink()
                rejected.append({
                    "image_id": image_id,
                    "reason": "exact_duplicate_sha256",
                    "sha256": digest,
                })
                continue
            (image_dir / f"{Path(filename).stem}.txt").write_text("", encoding="utf-8")
            selected_ids.add(image_id)
            selected_hashes.add(digest)
            count += 1
            selected.append({
                "kind": "negative",
                "distractor_class": class_name,
                "filename": filename,
                "image_id": image_id,
                "distractor_box_count": len(requested),
                "downloaded_url": used_url,
                "sha256": digest,
                **license_metadata(record),
            })
    return selected, rejected


def write_dataset_yaml(output):
    (output / "positive" / "dataset.yaml").write_text(
        "train: images/train\n\n"
        "names:\n"
        "  0: cup\n"
        "  1: mouse\n"
        "  2: keyboard\n"
        "  3: bottle\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    validate_args(args)
    output = prepare_output(args.output, args.overwrite)
    excluded_ids, excluded_hashes = existing_images_in(args.exclude_dir)
    positives, positive_rejections, selected_ids, selected_hashes = select_positives(
        output, excluded_ids, excluded_hashes, args
    )
    negatives, negative_rejections = select_negatives(
        output, selected_ids, selected_hashes, args
    )
    write_dataset_yaml(output)

    source_info = {
        "dataset": "Open Images V7 visualizer detection index",
        "index_base": INDEX_BASE,
        "annotations_license": "CC BY 4.0",
        "images_license": "Listed as CC BY 2.0; each selected record stores attribution metadata",
        "seed": args.seed,
        "requested": {
            "positive_per_class": args.positive_per_class,
            "mobile_phone": args.mobile_phone,
            "headphones": args.headphones,
            "power_plugs": args.power_plugs,
            "min_positive_box_area": args.min_positive_box_area,
            "min_distractor_box_area": args.min_distractor_box_area,
        },
        "excluded_existing_image_ids": len(excluded_ids),
        "excluded_existing_image_hashes": len(excluded_hashes),
        "selected": positives + negatives,
        "rejected": positive_rejections + negative_rejections,
    }
    (output / "source.json").write_text(
        json.dumps(source_info, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Positive candidates: {len(positives)}")
    print(f"Negative candidates: {len(negatives)}")
    print(f"Existing Open Images IDs excluded: {len(excluded_ids)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
