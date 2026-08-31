#!/usr/bin/env python3
"""用当前 YOLO 模型为 Open Images 双向增强候选排序并生成联系表。"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "cup", 1: "mouse", 2: "keyboard", 3: "bottle"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rank bidirectional candidates with the selected YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "external_openimages_bidirectional_candidates",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root.parent / "03_训练" / "weights" /
        "best_desktop_det_recollection_20260829.pt",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sheet-columns", type=int, default=4)
    parser.add_argument("--sheet-rows", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args):
    args.candidates = args.candidates.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.output = (
        args.output.expanduser().resolve()
        if args.output
        else args.candidates / "screening"
    )
    if not args.candidates.is_dir():
        raise SystemExit(f"Candidate directory does not exist: {args.candidates}")
    if not args.weights.is_file():
        raise SystemExit(f"Weights do not exist: {args.weights}")
    if not 0 < args.conf < 1 or not 0 < args.iou <= 1:
        raise SystemExit("--conf and --iou must be between zero and one.")
    if args.sheet_columns < 1 or args.sheet_rows < 1:
        raise SystemExit("Contact sheet dimensions must be positive.")
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output}. Use --overwrite.")
        shutil.rmtree(args.output)
    (args.output / "contact_sheets").mkdir(parents=True)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_images(root: Path):
    positive_dir = root / "positive" / "images" / "train"
    negative_dir = root / "negative_train"
    positives = sorted(
        path for path in positive_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    negatives = sorted(
        path for path in negative_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return positives, negatives


def load_yolo_labels(path: Path, width: int, height: int):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label in {path}: {line}")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        labels.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "box": [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ],
        })
    return labels


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


def predictions_from_result(result):
    predictions = []
    if result.boxes is None:
        return predictions
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    for box, class_id, confidence in zip(boxes, classes, confidences):
        class_id = int(class_id)
        predictions.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "confidence": round(float(confidence), 6),
            "box": [round(float(value), 2) for value in box],
        })
    return predictions


def analyze_positive(path, predictions, width, height, args):
    label_path = (
        args.candidates / "positive" / "labels" / "train" / f"{path.stem}.txt"
    )
    ground_truth = load_yolo_labels(label_path, width, height)
    matched_confidences = []
    missed = 0
    for truth in ground_truth:
        confidence = max(
            (
                prediction["confidence"]
                for prediction in predictions
                if prediction["class_id"] == truth["class_id"]
                and box_iou(prediction["box"], truth["box"]) >= args.iou
            ),
            default=0.0,
        )
        matched_confidences.append(confidence)
        if confidence < args.conf:
            missed += 1

    strong_predictions = [
        prediction for prediction in predictions
        if prediction["confidence"] >= args.conf
    ]
    unmatched_predictions = sum(
        not any(
            prediction["class_id"] == truth["class_id"]
            and box_iou(prediction["box"], truth["box"]) >= args.iou
            for truth in ground_truth
        )
        for prediction in strong_predictions
    )
    mean_match_confidence = (
        sum(matched_confidences) / len(matched_confidences)
        if matched_confidences else 0.0
    )
    difficulty = (
        2.0 * missed / max(1, len(ground_truth))
        + (1.0 - mean_match_confidence)
        + 0.2 * unmatched_predictions
    )
    requested_class = path.name.split("_", 1)[0]
    return {
        "kind": "positive",
        "path": str(path.relative_to(args.candidates)),
        "filename": path.name,
        "requested_class": requested_class,
        "width": width,
        "height": height,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "missed_ground_truth": missed,
        "unmatched_predictions": unmatched_predictions,
        "mean_match_confidence": round(mean_match_confidence, 6),
        "difficulty_score": round(difficulty, 6),
    }


def analyze_negative(path, predictions, width, height, args):
    strong_predictions = [
        prediction for prediction in predictions
        if prediction["confidence"] >= args.conf
    ]
    max_confidence = max(
        (prediction["confidence"] for prediction in predictions), default=0.0
    )
    distractor_class = path.name.rsplit("_", 1)[0]
    return {
        "kind": "negative",
        "path": str(path.relative_to(args.candidates)),
        "filename": path.name,
        "distractor_class": distractor_class,
        "width": width,
        "height": height,
        "ground_truth": [],
        "predictions": predictions,
        "false_positive_count": len(strong_predictions),
        "max_confidence": round(max_confidence, 6),
        "is_hard_negative": bool(strong_predictions),
        "difficulty_score": round(
            max_confidence + 0.1 * max(0, len(strong_predictions) - 1), 6
        ),
    }


def analyze_candidates(model, paths, positive_paths, args):
    records = []
    positive_set = {path.resolve() for path in positive_paths}
    predict_args = {
        "source": [str(path) for path in paths],
        "conf": 0.01,
        "imgsz": args.imgsz,
        "verbose": False,
        "stream": True,
    }
    if args.device:
        predict_args["device"] = args.device
    for result in model.predict(**predict_args):
        path = Path(result.path).resolve()
        height, width = result.orig_shape
        predictions = predictions_from_result(result)
        if path in positive_set:
            record = analyze_positive(path, predictions, width, height, args)
        else:
            record = analyze_negative(path, predictions, width, height, args)
        records.append(record)
    return records


def draw_box(image, box, color, label):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 2, cv2.LINE_AA,
    )


def annotated_image(record, root, conf):
    image = cv2.imread(str(root / record["path"]))
    if image is None:
        raise ValueError(f"Cannot read image: {record['path']}")
    for truth in record["ground_truth"]:
        draw_box(image, truth["box"], (30, 210, 30), f"GT {truth['class_name']}")
    for prediction in record["predictions"]:
        if prediction["confidence"] >= conf:
            draw_box(
                image,
                prediction["box"],
                (40, 40, 230),
                f"P {prediction['class_name']} {prediction['confidence']:.2f}",
            )
    return image


def fit_image(image, width, height):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = 255 * np.ones((height, width, 3), dtype="uint8")
    offset_x = (width - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
    return canvas


def make_contact_sheet(records, title, destination, args):
    tile_width = 360
    tile_height = 300
    header_height = 58
    page_size = args.sheet_columns * args.sheet_rows
    for page_index in range(0, len(records), page_size):
        page = records[page_index:page_index + page_size]
        sheet = 245 * np.ones(
            (args.sheet_rows * tile_height, args.sheet_columns * tile_width, 3),
            dtype="uint8",
        )
        for index, record in enumerate(page):
            row, column = divmod(index, args.sheet_columns)
            x = column * tile_width
            y = row * tile_height
            tile = fit_image(
                annotated_image(record, args.candidates, args.conf),
                tile_width,
                tile_height - header_height,
            )
            sheet[y + header_height:y + tile_height, x:x + tile_width] = tile
            if record["kind"] == "positive":
                summary = (
                    f"miss={record['missed_ground_truth']} "
                    f"match={record['mean_match_confidence']:.2f}"
                )
            else:
                summary = (
                    f"fp={record['false_positive_count']} "
                    f"max={record['max_confidence']:.2f}"
                )
            lines = [f"{page_index + index + 1}. {record['filename']}", summary]
            for line_index, line in enumerate(lines):
                cv2.putText(
                    sheet, line[:48], (x + 7, y + 21 + 25 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1,
                    cv2.LINE_AA,
                )
            cv2.rectangle(
                sheet, (x, y), (x + tile_width - 1, y + tile_height - 1),
                (160, 160, 160), 1,
            )
        page_number = page_index // page_size + 1
        output = destination / f"{title}_{page_number:02d}.jpg"
        cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])


def write_contact_sheets(records, args):
    destination = args.output / "contact_sheets"
    positives = defaultdict(list)
    negatives = defaultdict(list)
    for record in records:
        if record["kind"] == "positive":
            positives[record["requested_class"]].append(record)
        else:
            negatives[record["distractor_class"]].append(record)
    for class_name, class_records in positives.items():
        class_records.sort(key=lambda item: item["difficulty_score"], reverse=True)
        make_contact_sheet(class_records, f"positive_{class_name}", destination, args)
    for class_name, class_records in negatives.items():
        class_records.sort(key=lambda item: item["difficulty_score"], reverse=True)
        make_contact_sheet(class_records, f"negative_{class_name}", destination, args)


def summarize(records):
    positives = [record for record in records if record["kind"] == "positive"]
    negatives = [record for record in records if record["kind"] == "negative"]
    return {
        "positive_images": len(positives),
        "positive_missed_images": sum(
            record["missed_ground_truth"] > 0 for record in positives
        ),
        "negative_images": len(negatives),
        "hard_negative_images": sum(
            record["is_hard_negative"] for record in negatives
        ),
        "hard_negatives_by_distractor": {
            class_name: sum(
                record["is_hard_negative"]
                for record in negatives
                if record["distractor_class"] == class_name
            )
            for class_name in sorted({record["distractor_class"] for record in negatives})
        },
    }


def main():
    args = parse_args()
    validate_args(args)
    positive_paths, negative_paths = candidate_images(args.candidates)
    if not positive_paths and not negative_paths:
        raise SystemExit("No candidate images found.")

    model = YOLO(str(args.weights))
    model_names = {int(class_id): name for class_id, name in model.names.items()}
    if model_names != CLASS_NAMES:
        raise SystemExit(
            f"Unexpected model classes: {model_names}; expected {CLASS_NAMES}"
        )
    records = analyze_candidates(
        model, positive_paths + negative_paths, positive_paths, args
    )
    records.sort(key=lambda item: (item["kind"], item["path"]))
    write_contact_sheets(records, args)
    report = {
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "candidate_root": str(args.candidates),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "image_size": args.imgsz,
        "summary": summarize(records),
        "records": records,
    }
    (args.output / "screening.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output / 'screening.json'}")
    print(f"Contact sheets: {args.output / 'contact_sheets'}")


if __name__ == "__main__":
    main()
