"""验收测试脚本：测 20 个物体的识别正确率、Jetson 实时 FPS，并保存错误案例。

两种模式:
  1) 逐物体测试（验收主流程）—— 每次摆一个物体，输入真值类别，按空格抓拍判定
        python3 eval_acceptance.py --weights best.engine --classes cup mouse keyboard bottle
  2) 纯 FPS 基准 —— 连续跑 200 帧统计推理速度
        python3 eval_acceptance.py --weights best.engine --fps-only

产物:
    results/summary.json     正确率、每类统计、FPS
    results/records.csv      20 次测试的逐条记录
    results/errors/*.jpg     识别错误的典型案例（图上标注了真值和预测）
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
    # UVC 摄像头默认 YUYV 未压缩，720p 下只能给 5~10 FPS，必须先切 MJPG 再设分辨率
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_fps_benchmark(model: YOLO, cap: cv2.VideoCapture, imgsz: int, n: int = 200) -> float:
    """预热 20 帧后统计纯推理耗时，避免首帧 CUDA 初始化拉低数字。"""
    for _ in range(20):
        ok, frame = cap.read()
        if ok:
            model.predict(frame, imgsz=imgsz, verbose=False)

    times = []
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        model.predict(frame, imgsz=imgsz, verbose=False)
        times.append(time.time() - t0)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} 帧...")

    fps = len(times) / sum(times)
    print(f"\n平均推理 FPS: {fps:.2f}  (单帧 {1000 * sum(times) / len(times):.1f} ms)")
    return fps


def main() -> None:
    parser = argparse.ArgumentParser(description="目标检测验收测试")
    parser.add_argument("--weights", default="best.engine")
    parser.add_argument("--source", default="0")
    parser.add_argument("--classes", nargs="+", required=False, help="类别名列表，顺序须与训练一致")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trials", type=int, default=20, help="测试物体个数")
    parser.add_argument("--fps-only", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    names = list(args.classes) if args.classes else list(model.names.values())
    cap = open_camera(args.source, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头: {args.source}")

    OUT_DIR.mkdir(exist_ok=True)
    err_dir = OUT_DIR / "errors"
    err_dir.mkdir(exist_ok=True)

    try:
        if args.fps_only:
            fps = run_fps_benchmark(model, cap, args.imgsz)
            (OUT_DIR / "fps_benchmark.json").write_text(
                json.dumps({"fps": round(fps, 2), "time": datetime.now().isoformat()}, indent=2)
            )
            return

        print("\n先做 FPS 基准测试...")
        fps = run_fps_benchmark(model, cap, args.imgsz, n=100)

        print("\n===== 逐物体测试 =====")
        print(f"类别: {', '.join(f'{i}:{n}' for i, n in enumerate(names))}")
        print("摆好物体后按 [空格] 抓拍判定，[q] 提前结束\n")

        records = []
        while len(records) < args.trials:
            idx = len(records) + 1
            raw = input(f"[{idx}/{args.trials}] 真值类别（输名字或编号，q 结束）: ").strip()
            if raw.lower() == "q":
                break
            truth = names[int(raw)] if raw.isdigit() and int(raw) < len(names) else raw
            if truth not in names:
                print(f"  未知类别，可选: {names}")
                continue

            # 冲掉摄像头缓冲，确保拿到摆好之后的画面
            frame = None
            for _ in range(5):
                ok, frame = cap.read()
            if frame is None:
                print("  抓帧失败")
                continue

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
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
            print(f"  预测: {pred} ({score:.2f})  ->  {'✅ 正确' if correct else '❌ 错误'}")

            annotated = result.plot()
            cv2.putText(annotated, f"GT: {truth} | PRED: {pred} {score:.2f}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if correct else (0, 0, 255), 2)
            if not correct:
                cv2.imwrite(str(err_dir / f"error_{idx:02d}_{truth}_as_{pred}.jpg"), annotated)
            cv2.imshow("result", annotated)
            cv2.waitKey(600)

        if not records:
            print("没有任何测试记录")
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

        print("\n===== 验收结果 =====")
        print(f"正确率 : {n_correct}/{len(records)} = {acc:.1%}   "
              f"{'✅ 达标(≥80%)' if acc >= 0.8 else '❌ 未达标'}")
        print(f"FPS    : {fps:.2f}   {'✅ 达标(≥5)' if fps >= 5 else '❌ 未达标'}")
        for cls, st in per_class.items():
            print(f"  {cls:10s} {st['correct']}/{st['total']}")
        if confusion:
            print("典型错误:")
            for (t, p), c in confusion.most_common():
                print(f"  {t} 被识别成 {p}  x{c}")
        print(f"\n结果已保存到 {OUT_DIR}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
