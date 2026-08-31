#!/usr/bin/env python3
"""逐图评估干扰物误报与灰色保温瓶召回。"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="评估目标检测模型的定向回归场景")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--pure-negative-dir", type=Path, action="append", default=[])
    parser.add_argument("--mixed-image", type=Path)
    parser.add_argument("--headphones-dir", type=Path, required=True)
    parser.add_argument("--thermos-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--low-conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(directories):
    images = []
    for directory in directories:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise SystemExit(f"图片目录不存在: {directory}")
        images.extend(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(images)


def predictions_for(model, image, args, confidence):
    result = model.predict(
        str(image), conf=confidence, iou=args.iou, imgsz=args.imgsz,
        device=args.device, verbose=False,
    )[0]
    predictions = []
    for class_id, score, box in zip(
        result.boxes.cls, result.boxes.conf, result.boxes.xyxy
    ):
        predictions.append({
            "class": result.names[int(class_id)],
            "confidence": round(float(score), 6),
            "xyxy": [round(float(value), 2) for value in box],
        })
    return predictions


def evaluate_pure_negatives(model, images, args):
    details = {}
    class_counts = Counter()
    max_confidence = 0.0
    for image in images:
        predictions = predictions_for(model, image, args, args.conf)
        if predictions:
            details[image.name] = predictions
        class_counts.update(item["class"] for item in predictions)
        max_confidence = max(
            max_confidence,
            max((item["confidence"] for item in predictions), default=0.0),
        )
    return {
        "images": len(images),
        "false_positive_images": len(details),
        "false_positive_predictions": sum(class_counts.values()),
        "predictions_by_class": dict(sorted(class_counts.items())),
        "max_confidence": round(max_confidence, 6),
        "details": details,
    }


def evaluate_headphones(model, images, args):
    details = {}
    mouse_confidences = []
    for image in images:
        predictions = predictions_for(model, image, args, args.conf)
        mouse_predictions = [
            item for item in predictions if item["class"] == "mouse"
        ]
        if mouse_predictions:
            details[image.name] = mouse_predictions
            mouse_confidences.extend(
                item["confidence"] for item in mouse_predictions
            )
    return {
        "images": len(images),
        "frames_with_mouse_false_positive": len(details),
        "mouse_false_positive_predictions": len(mouse_confidences),
        "max_mouse_confidence": round(max(mouse_confidences, default=0.0), 6),
        "details": details,
    }


def evaluate_thermos(model, image, args):
    predictions = predictions_for(model, image, args, args.low_conf)
    best = Counter()
    for prediction in predictions:
        best[prediction["class"]] = max(
            best[prediction["class"]], prediction["confidence"]
        )
    bottle_confidence = best["bottle"]
    return {
        "image": image.name,
        "bottle_confidence": round(bottle_confidence, 6),
        "cup_confidence": round(best["cup"], 6),
        "passes_bottle_gate": bottle_confidence >= args.conf,
        "low_threshold_predictions": predictions,
    }


def main():
    args = parse_args()
    if not 0 < args.low_conf <= args.conf < 1:
        raise SystemExit("应满足 0 < --low-conf <= --conf < 1")
    if not 0 < args.iou <= 1 or args.imgsz < 1:
        raise SystemExit("--iou 或 --imgsz 非法")

    models = [path.expanduser().resolve() for path in args.model]
    for path in models:
        if not path.is_file():
            raise SystemExit(f"模型不存在: {path}")
    pure_negatives = collect_images(args.pure_negative_dir)
    headphones = collect_images([args.headphones_dir])
    thermos = args.thermos_image.expanduser().resolve()
    mixed = args.mixed_image.expanduser().resolve() if args.mixed_image else None
    for path in [thermos, mixed]:
        if path is not None and not path.is_file():
            raise SystemExit(f"评估图片不存在: {path}")

    report = {
        "protocol": {
            "confidence": args.conf,
            "low_confidence": args.low_conf,
            "nms_iou": args.iou,
            "image_size": args.imgsz,
            "device": args.device,
            "inference": "one image at a time",
        },
        "models": {},
    }
    for model_path in models:
        model = YOLO(str(model_path))
        report["models"][model_path.stem] = {
            "weights_sha256": sha256_file(model_path),
            "pure_negative": evaluate_pure_negatives(
                model, pure_negatives, args
            ),
            "mixed_regression": (
                predictions_for(model, mixed, args, args.conf) if mixed else None
            ),
            "headphones_sequence": evaluate_headphones(
                model, headphones, args
            ),
            "gray_thermos": evaluate_thermos(model, thermos, args),
        }

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        name: {
            "negative_fp_images": result["pure_negative"]["false_positive_images"],
            "headphones_mouse_frames": result["headphones_sequence"]["frames_with_mouse_false_positive"],
            "thermos_bottle_confidence": result["gray_thermos"]["bottle_confidence"],
        }
        for name, result in report["models"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()