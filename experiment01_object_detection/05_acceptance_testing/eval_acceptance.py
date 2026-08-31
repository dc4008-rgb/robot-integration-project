"""Acceptance test for recognition accuracy on 20 objects and real-time Jetson FPS.

The script also saves error cases and supports two modes:
    1) Per-object test (main acceptance workflow) - present one object at a time,
         enter its ground-truth class, and press Space to capture and evaluate it
        python3 eval_acceptance.py --weights best.engine --classes cup mouse keyboard bottle
    2) FPS-only benchmark - process 200 consecutive frames to measure inference speed
        python3 eval_acceptance.py --weights best.engine --fps-only

Outputs:
        results/summary.json     Accuracy, per-class statistics, and FPS
        results/records.csv      Individual records for the 20 trials
        results/errors/*.jpg     Representative errors annotated with ground truth and prediction
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

OUT_DIR = Path(__file__).resolve().parent / "results"


def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    if source == "csi":
        pipeline = (
            f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    # UVC cameras default to uncompressed YUYV, which provides only 5-10 FPS at 720p.
    # Switch to MJPG before setting the resolution.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_fps_benchmark(
    model: YOLO, cap: cv2.VideoCapture, imgsz: int, iou: float, n: int = 200
) -> float:
    """Warm up for 20 frames, then measure inference time without CUDA startup skew."""
    for _ in range(20):
        ok, frame = cap.read()
        if ok:
            model.predict(frame, imgsz=imgsz, iou=iou, verbose=False)

    times = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        model.predict(frame, imgsz=imgsz, iou=iou, verbose=False)
        times.append(time.time() - t0)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} frames...")

    fps = len(times) / sum(times)
    print(f"\nAverage inference FPS: {fps:.2f}  ({1000 * sum(times) / len(times):.1f} ms per frame)")
    return fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Object detection acceptance test")
    parser.add_argument("--weights", default="best.engine")
    parser.add_argument("--source", default="0")
    parser.add_argument("--classes", nargs="+", required=False, help="Class names in training order")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trials", type=int, default=20, help="Number of test objects")
    parser.add_argument("--fps-only", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    names = list(args.classes) if args.classes else list(model.names.values())
    cap = open_camera(args.source, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open camera: {args.source}")

    OUT_DIR.mkdir(exist_ok=True)
    err_dir = OUT_DIR / "errors"
    err_dir.mkdir(exist_ok=True)

    try:
        if args.fps_only:
            fps = run_fps_benchmark(model, cap, args.imgsz, args.iou)
            (OUT_DIR / "fps_benchmark.json").write_text(
                json.dumps({"fps": round(fps, 2), "time": datetime.now().isoformat()}, indent=2)
            )
            return

        print("\nRunning FPS benchmark first...")
        fps = run_fps_benchmark(model, cap, args.imgsz, args.iou, n=100)

        print("\n===== Per-object test =====")
        print(f"Classes: {', '.join(f'{i}:{n}' for i, n in enumerate(names))}")
        print("Place the object, then press [Space] to capture and evaluate; press [q] to stop early\n")

        records = []
        while len(records) < args.trials:
            idx = len(records) + 1
            raw = input(f"[{idx}/{args.trials}] Ground-truth class (name or index; q to quit): ").strip()
            if raw.lower() == "q":
                break
            truth = names[int(raw)] if raw.isdigit() and int(raw) < len(names) else raw
            if truth not in names:
                print(f"  Unknown class. Available classes: {names}")
                continue

            # Flush the camera buffer to capture a frame after the object is positioned.
            frame = None
            for _ in range(5):
                ok, frame = cap.read()
            if frame is None:
                print("  Failed to capture frame")
                continue

            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)
            if boxes:
                top = boxes[0]
                pred = model.names[int(top.cls[0])]
                score = float(top.conf[0])
            else:
                pred, score = "none", 0.0

            correct = pred == truth
            records.append({"idx": idx, "truth": truth, "pred": pred,
                            "score": round(score, 3), "correct": int(correct)})
            print(f"  Prediction: {pred} ({score:.2f})  ->  {'PASS' if correct else 'FAIL'}")

            annotated = result.plot()
            cv2.putText(annotated, f"GT: {truth} | PRED: {pred} {score:.2f}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if correct else (0, 0, 255), 2)
            if not correct:
                cv2.imwrite(str(err_dir / f"error_{idx:02d}_{truth}_as_{pred}.jpg"), annotated)
            cv2.imshow("result", annotated)
            cv2.waitKey(600)

        if not records:
            print("No test records were collected")
            return

        n_correct = sum(r["correct"] for r in records)
        acc = n_correct / len(records)

        per_class = defaultdict(lambda: {"total": 0, "correct": 0})
        confusion = Counter()
        for r in records:
            per_class[r["truth"]]["total"] += 1
            per_class[r["truth"]]["correct"] += r["correct"]
            if not r["correct"]:
                confusion[(r["truth"], r["pred"])] += 1

        summary = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "weights": args.weights,
            "conf_threshold": args.conf,
            "nms_iou_threshold": args.iou,
            "total": len(records),
            "correct": n_correct,
            "accuracy": round(acc, 4),
            "fps": round(fps, 2),
            "pass_accuracy": acc >= 0.8,
            "pass_fps": fps >= 5.0,
            "per_class": {k: dict(v) for k, v in per_class.items()},
            "confusions": [{"truth": t, "pred": p, "count": c}
                           for (t, p), c in confusion.most_common()],
        }
        (OUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (OUT_DIR / "records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "truth", "pred", "score", "correct"])
            writer.writeheader()
            writer.writerows(records)

        print("\n===== Acceptance results =====")
        print(f"Accuracy: {n_correct}/{len(records)} = {acc:.1%}   "
              f"{'PASS (>=80%)' if acc >= 0.8 else 'FAIL'}")
        print(f"FPS     : {fps:.2f}   {'PASS (>=5)' if fps >= 5 else 'FAIL'}")
        for cls, st in per_class.items():
            print(f"  {cls:10s} {st['correct']}/{st['total']}")
        if confusion:
            print("Representative errors:")
            for (t, p), c in confusion.most_common():
                print(f"  {t} predicted as {p}  x{c}")
        print(f"\nResults saved to {OUT_DIR}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
