#!/usr/bin/env python3
"""从两批 Open Images 候选中筛选四类平衡正样本。"""
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["cup", "mouse", "keyboard", "bottle"]
EXPECTED_MODEL_SHA256 = (
    "1c734e9738a11353f9cd7467cdf5694892408578bb0312f291c5fc2e31f33ed2"
)

VISUAL_EXCLUSIONS = set("""
cup_1a1b577feb52a707.jpg
cup_9eae7928772a2db9.jpg
cup_96872c9ed02d0e78.jpg
cup_00d64443ebc47f58.jpg
cup_a4e9c65bd308629f.jpg
cup_d14412be47f4da71.jpg
cup_fffd0267b66bba49.jpg
cup_24dd5539eb6a6546.jpg
cup_34c4e5b123e9f33f.jpg
cup_4915de566fce4430.jpg
cup_a6960906a76ad852.jpg
cup_d46fb0077f31cf1f.jpg
cup_1088928dd27d2299.jpg
cup_00a76c83fc62f17e.jpg
cup_0001f65de725a864.jpg
cup_364cdb2d4b4da5f2.jpg
cup_01d20a8099683813.jpg
cup_06835a1e7f04dac0.jpg
cup_0491e739a2eac5ea.jpg
cup_25b182c123d2efc9.jpg
cup_100940397939f11e.jpg
cup_24f0153687071429.jpg
cup_10a8071480a33c81.jpg
cup_e87f9ae6342b5d45.jpg
cup_016ec667545f3cd8.jpg
cup_5b996e1eef020312.jpg
cup_2b334b4b86c4b0ff.jpg
cup_b1d2a9b92ee88d99.jpg
cup_897ec959e3a87d8c.jpg
cup_464b7084ad087e0a.jpg
cup_0180a26da819ea7e.jpg
cup_0b57ca6e2c4fe3d7.jpg
cup_f391a3122bbe494c.jpg
cup_013d411cce8c0daa.jpg
cup_1f250b4532ae9438.jpg
cup_37205d624f4ca8dd.jpg
cup_f5cfa970674dd724.jpg
cup_ff04e512febf08dd.jpg
cup_266817e46a1eee31.jpg
cup_0fbb1a2319167151.jpg
cup_85e407a42d86defc.jpg
mouse_2c6ba58cf7a93155.jpg
mouse_31f0f83da15b8a2d.jpg
mouse_438c6c30fc036bd0.jpg
mouse_a6f2e14f97290078.jpg
mouse_f649a6e9825f29c9.jpg
mouse_2f3a5d6a791456cc.jpg
mouse_6d357f8410328fe4.jpg
mouse_22a267a78fcdab4b.jpg
mouse_3267fc6a62e95092.jpg
mouse_34aae7b2696900f5.jpg
mouse_3d40f63c26d1764c.jpg
mouse_4a957d0b5014fe81.jpg
mouse_6b9a0b35d3811bd0.jpg
mouse_e043920f554972ef.jpg
mouse_d7a3c5733edb457d.jpg
mouse_e79b59a8586f9a02.jpg
mouse_7a55dc2e32ad3569.jpg
mouse_4e76908aa4159087.jpg
mouse_1c6d3da0629ce39f.jpg
mouse_742869228c83d4c4.jpg
mouse_11b5cce32ed91e2a.jpg
mouse_5aaca29fee4a8c2b.jpg
mouse_32b58a87afdeee70.jpg
mouse_42c22d0bcf2f54ed.jpg
mouse_1689707b752888a7.jpg
mouse_5ddf420602ad187f.jpg
mouse_2dd89a0ac8f9d12a.jpg
mouse_51378edccc6b7afa.jpg
mouse_82b4071b49ec8045.jpg
mouse_a1b7fdb5b1615b69.jpg
mouse_569c5142100311c9.jpg
mouse_8804cdd92802f671.jpg
mouse_0c5dd6b194d2d3f3.jpg
mouse_bfed674627dc9b9c.jpg
mouse_3a1fd03ca8b3ece8.jpg
mouse_1fc6c1e99988d314.jpg
mouse_28ed875cfc06a7a8.jpg
mouse_532dfda06bfe8873.jpg
mouse_5b909d92a8e958ed.jpg
mouse_3f2ae9d9771af9fc.jpg
mouse_b027f9746d980500.jpg
mouse_41db740dad62d00c.jpg
mouse_0600da870728676a.jpg
mouse_10c1adeecfd4948c.jpg
mouse_594f7b26f02986a8.jpg
mouse_c3b3d8897640c70e.jpg
mouse_3bac9b649f7f6896.jpg
mouse_461a58b5c11b3a1e.jpg
mouse_5915f81b541cceb5.jpg
mouse_5948df8916da65b7.jpg
mouse_6ab6fce73c12a861.jpg
mouse_6fdc43a1e1382636.jpg
mouse_98364902bb658faa.jpg
mouse_b00e326effb20842.jpg
mouse_b2b683614cf45fc1.jpg
mouse_bc40a7f386a7d743.jpg
mouse_08b740755636fc1b.jpg
mouse_4d5f67c332f2d5e9.jpg
mouse_3ff79abeccb7c4b3.jpg
mouse_01e47f826eecc121.jpg
mouse_0f126a8a40a3ecfc.jpg
mouse_8de5a9de5de6da3a.jpg
mouse_0517e96ceef0a327.jpg
mouse_553dacea0b35df28.jpg
mouse_d5ae4b85afe82da6.jpg
mouse_78a5ea362a254281.jpg
mouse_528deebb2c7fe34e.jpg
mouse_0bb525427f9dedc3.jpg
mouse_3f5ad9a1210a0752.jpg
mouse_59430d1b83477d65.jpg
mouse_a21a38828a9d3b39.jpg
mouse_324f9aab6ff2d5bb.jpg
mouse_28f618a7e62188b2.jpg
keyboard_b4946ed99c465718.jpg
keyboard_3bd628328704a2f5.jpg
keyboard_a6b60750384df9f5.jpg
keyboard_83cd1030bb174730.jpg
keyboard_9ff5c43ad942f775.jpg
keyboard_081a9e3484226707.jpg
bottle_3733c584ce1f21c5.jpg
bottle_e2b3c1a87966781c.jpg
bottle_2946a21a48a238e0.jpg
bottle_9dbdad202f5a7b7e.jpg
bottle_cd3c718b750e4b25.jpg
bottle_7aebea888e2ca0d1.jpg
bottle_05f4120cf681e6cc.jpg
bottle_ec6e4bec411c69ca.jpg
bottle_b49dd7f7d3765931.jpg
bottle_482968a0b878f24c.jpg
bottle_c9198a9134a9e559.jpg
bottle_05accc3fd24b21d9.jpg
""".split())


def parse_args():
    root = Path(__file__).resolve().parent
    candidates = root / "external_openimages_bidirectional_candidates"
    parser = argparse.ArgumentParser(
        description="Curate balanced Open Images positives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        action="append",
        default=[
            candidates / "balanced_positive_20260831",
            candidates / "balanced_positive_supplement_20260831",
        ],
    )
    parser.add_argument(
        "--exclude-dir",
        type=Path,
        action="append",
        default=[root / "images"],
        help="Existing image trees used for perceptual deduplication.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "external_openimages_balanced_100_yolo",
    )
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--hard-ratio", type=float, default=0.55)
    parser.add_argument("--min-box-area", type=float, default=0.025)
    parser.add_argument("--min-sharpness", type=float, default=12.0)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.per_class < 1:
        raise SystemExit("--per-class must be positive.")
    if not 0 <= args.hard_ratio <= 1:
        raise SystemExit("--hard-ratio must be in [0, 1].")
    if not 0 < args.min_box_area <= 1:
        raise SystemExit("--min-box-area must be in (0, 1].")
    if args.min_sharpness < 0 or args.phash_distance < 0:
        raise SystemExit("Quality and hash thresholds must be non-negative.")


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(image):
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(np.float32(resized))[:8, :8]
    values = transformed.flatten()[1:]
    median = float(np.median(values))
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(first, second):
    return (first ^ second).bit_count()


def image_metrics(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Cannot decode image: {path}")
    return {
        "width": image.shape[1],
        "height": image.shape[0],
        "sharpness": float(cv2.Laplacian(image, cv2.CV_64F).var()),
        "phash": phash(image),
    }


def box_area(box, width, height):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (
        width * height
    )


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def has_duplicate_truth(truths):
    for index, first in enumerate(truths):
        for second in truths[index + 1:]:
            if (
                first["class_id"] == second["class_id"]
                and box_iou(first["box"], second["box"]) >= 0.8
            ):
                return True
    return False


def has_extreme_crop(truths, requested_class, width, height):
    for truth in truths:
        if truth["class_name"] != requested_class:
            continue
        left, top, right, bottom = truth["box"]
        border_count = sum((left <= 1, top <= 1, right >= width - 1, bottom >= height - 1))
        area = box_area(truth["box"], width, height)
        if border_count >= 2 or (border_count and area > 0.55):
            return True
    return False


def existing_phashes(paths):
    hashes = []
    for root in paths:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Exclude directory does not exist: {root}")
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                hashes.append(phash(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)))
    return hashes


def load_candidates(args, baseline_hashes):
    records = []
    source_hashes = set()
    for root in args.candidates:
        root = root.expanduser().resolve()
        source_path = root / "source.json"
        screening_path = root / "screening" / "screening.json"
        if not source_path.is_file() or not screening_path.is_file():
            raise SystemExit(f"Incomplete candidate root: {root}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        screening = json.loads(screening_path.read_text(encoding="utf-8"))
        if screening.get("weights_sha256") != EXPECTED_MODEL_SHA256:
            raise SystemExit(f"Unexpected screening model for {root}")
        source_by_name = {item["filename"]: item for item in source["selected"]}

        for screening_record in screening["records"]:
            filename = screening_record["filename"]
            source_record = source_by_name.get(filename)
            if source_record is None:
                raise SystemExit(f"Missing source record: {filename}")
            if source_record["sha256"] in source_hashes:
                raise SystemExit(f"Duplicate candidate content hash: {filename}")
            source_hashes.add(source_record["sha256"])
            image_path = root / "positive" / "images" / "train" / filename
            label_path = root / "positive" / "labels" / "train" / f"{Path(filename).stem}.txt"
            if sha256_file(image_path) != source_record["sha256"]:
                raise SystemExit(f"Source image hash mismatch: {image_path}")
            metrics = image_metrics(image_path)
            reasons = []
            requested_class = screening_record["requested_class"]
            truths = screening_record["ground_truth"]
            requested_truths = [
                truth for truth in truths if truth["class_name"] == requested_class
            ]
            if filename in VISUAL_EXCLUSIONS:
                reasons.append("visual_audit_exclusion")
            if screening_record["unmatched_predictions"]:
                reasons.append("possible_unlabeled_target")
            if not requested_truths:
                reasons.append("missing_requested_truth")
            maximum_area = max(
                (
                    box_area(truth["box"], metrics["width"], metrics["height"])
                    for truth in requested_truths
                ),
                default=0.0,
            )
            if maximum_area < args.min_box_area:
                reasons.append("requested_target_too_small")
            if has_duplicate_truth(truths):
                reasons.append("duplicate_ground_truth")
            if has_extreme_crop(
                truths, requested_class, metrics["width"], metrics["height"]
            ):
                reasons.append("extreme_target_crop")
            if metrics["sharpness"] < args.min_sharpness:
                reasons.append("low_sharpness")
            nearest_existing = min(
                (hamming(metrics["phash"], item) for item in baseline_hashes),
                default=63,
            )
            if nearest_existing <= args.phash_distance:
                reasons.append("near_existing_image")

            records.append({
                "filename": filename,
                "requested_class": requested_class,
                "candidate_batch": root.name,
                "image_path": image_path,
                "label_path": label_path,
                "sha256": source_record["sha256"],
                "source": source_record,
                "screening": screening_record,
                "sharpness": round(metrics["sharpness"], 3),
                "phash": metrics["phash"],
                "nearest_existing_phash_distance": nearest_existing,
                "maximum_requested_box_area": round(maximum_area, 6),
                "rejection_reasons": sorted(set(reasons)),
            })
    return records


def select_records(records, args):
    selected = []
    selected_hashes = []
    by_class = defaultdict(list)
    for record in records:
        if not record["rejection_reasons"]:
            by_class[record["requested_class"]].append(record)

    for class_name in CLASS_NAMES:
        eligible = by_class[class_name]
        hard = sorted(
            (item for item in eligible if item["screening"]["missed_ground_truth"]),
            key=lambda item: (-item["screening"]["difficulty_score"], item["filename"]),
        )
        routine = sorted(
            (item for item in eligible if not item["screening"]["missed_ground_truth"]),
            key=lambda item: (-item["screening"]["difficulty_score"], item["filename"]),
        )
        desired_hard = min(round(args.per_class * args.hard_ratio), len(hard))
        ordered = hard[:desired_hard] + routine + hard[desired_hard:]
        class_selected = []
        for record in ordered:
            nearest_selected = min(
                (hamming(record["phash"], item) for item in selected_hashes),
                default=63,
            )
            if nearest_selected <= args.phash_distance:
                record["rejection_reasons"] = ["near_selected_image"]
                continue
            record["nearest_selected_phash_distance"] = nearest_selected
            class_selected.append(record)
            selected_hashes.append(record["phash"])
            if len(class_selected) == args.per_class:
                break
        if len(class_selected) != args.per_class:
            raise SystemExit(
                f"Only {len(class_selected)} eligible {class_name} images remain; "
                f"need {args.per_class}."
            )
        selected.extend(class_selected)
    return selected


def prepare_output(output: Path, overwrite: bool):
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    (output / "images" / "train").mkdir(parents=True)
    (output / "labels" / "train").mkdir(parents=True)


def materialize(selected, output):
    for record in selected:
        image_destination = output / "images" / "train" / record["filename"]
        label_destination = output / "labels" / "train" / record["label_path"].name
        shutil.copy2(record["image_path"], image_destination)
        shutil.copy2(record["label_path"], label_destination)
        if sha256_file(image_destination) != record["sha256"]:
            raise SystemExit(f"Output hash mismatch: {image_destination}")
    (output / "dataset.yaml").write_text(
        "train: images/train\n\n"
        "names:\n"
        "  0: cup\n"
        "  1: mouse\n"
        "  2: keyboard\n"
        "  3: bottle\n",
        encoding="utf-8",
    )


def public_record(record):
    return {
        key: value
        for key, value in record.items()
        if key not in {"image_path", "label_path", "phash"}
    } | {"phash_hex": f"{record['phash']:016x}"}


def main():
    args = parse_args()
    validate_args(args)
    output = args.output.expanduser().resolve()
    baseline_hashes = existing_phashes(args.exclude_dir)
    records = load_candidates(args, baseline_hashes)
    selected = select_records(records, args)
    prepare_output(output, args.overwrite)
    materialize(selected, output)

    selected_names = {record["filename"] for record in selected}
    selected_by_class = Counter(record["requested_class"] for record in selected)
    hard_by_class = Counter(
        record["requested_class"]
        for record in selected
        if record["screening"]["missed_ground_truth"]
    )
    box_counts = Counter()
    for record in selected:
        for truth in record["screening"]["ground_truth"]:
            box_counts[truth["class_name"]] += 1
    rejection_counts = Counter(
        reason
        for record in records
        if record["filename"] not in selected_names
        for reason in record["rejection_reasons"]
    )
    manifest = {
        "selection_version": 1,
        "candidate_roots": [str(path.expanduser().resolve()) for path in args.candidates],
        "screening_model_sha256": EXPECTED_MODEL_SHA256,
        "criteria": {
            "per_requested_class": args.per_class,
            "hard_ratio_target": args.hard_ratio,
            "minimum_requested_box_area": args.min_box_area,
            "minimum_laplacian_sharpness": args.min_sharpness,
            "maximum_phash_distance_for_rejection": args.phash_distance,
            "reject_unmatched_model_predictions": True,
            "reject_duplicate_ground_truth_iou": 0.8,
            "visual_exclusion_count": len(VISUAL_EXCLUSIONS),
        },
        "counts": {
            "candidate_images": len(records),
            "selected_images": len(selected),
            "selected_by_requested_class": {
                name: selected_by_class[name] for name in CLASS_NAMES
            },
            "hard_images_by_requested_class": {
                name: hard_by_class[name] for name in CLASS_NAMES
            },
            "selected_box_counts": {
                name: box_counts[name] for name in CLASS_NAMES
            },
            "rejections_by_reason": dict(sorted(rejection_counts.items())),
        },
        "selected": [public_record(record) for record in selected],
    }
    (output / "selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()