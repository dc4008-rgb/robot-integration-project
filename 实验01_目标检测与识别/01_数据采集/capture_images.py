"""摄像头采集脚本：在 Jetson 或本地电脑上采集训练图片。

用法:
    python capture_images.py --tag cup                    # 按 s 保存一张，q 退出
    python capture_images.py --tag mixed --interval 0.5   # 每 0.5 秒自动存一张
    python capture_images.py --source csi                 # Jetson CSI 摄像头
    python capture_images.py --tag mouse --output-dir data/train

SSH 远程操作（无图形界面）时必须用无窗口模式：
    python capture_images.py --tag cup --no-display --interval 0.5 --count 60
"""
import argparse
import csv
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2

RAW_DIR = Path(__file__).resolve().parent.parent / "02_数据集" / "raw"


def csi_pipeline(width: int, height: int, fps: int) -> str:
    """Jetson CSI 摄像头（IMX219 等）的 GStreamer 管线。"""
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )


def open_camera(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if source == "csi":
        return cv2.VideoCapture(csi_pipeline(width, height, fps), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    # UVC 摄像头默认 YUYV 未压缩，720p 下带宽跑满只能给 5~10 FPS，必须先切 MJPG 再设分辨率
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def configure_uvc_camera(cap: cv2.VideoCapture, source: str, args) -> None:
    source_index = int(source) if source.isdigit() else None
    executable = shutil.which("v4l2-ctl")
    if source_index is not None and executable is not None:
        auto_exposure = args.auto_exposure
        if auto_exposure == 0.25:
            auto_exposure = 1
        elif auto_exposure == 0.75:
            auto_exposure = 3
        controls = [
            ("auto_exposure", auto_exposure),
            ("exposure_time_absolute", args.exposure),
            ("gain", args.gain),
            ("backlight_compensation", args.backlight_compensation),
            ("gamma", args.gamma),
            ("exposure_dynamic_framerate", 0 if args.fixed_camera_fps else None),
        ]
        for name, value in controls:
            if value is None:
                continue
            result = subprocess.run(
                [
                    executable,
                    "-d", f"/dev/video{source_index}",
                    f"--set-ctrl={name}={value:g}",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                message = result.stderr.strip() or result.stdout.strip()
                print(f"警告: 摄像头不接受 {name}={value:g}: {message}")
            else:
                print(f"摄像头 {name}={value:g}")
        return

    settings = [
        (cv2.CAP_PROP_AUTO_EXPOSURE, args.auto_exposure, "自动曝光"),
        (cv2.CAP_PROP_EXPOSURE, args.exposure, "曝光"),
        (cv2.CAP_PROP_GAIN, args.gain, "增益"),
        (cv2.CAP_PROP_BACKLIGHT, args.backlight_compensation, "背光补偿"),
        (cv2.CAP_PROP_GAMMA, args.gamma, "Gamma"),
    ]
    for property_id, value, name in settings:
        if value is not None and not cap.set(property_id, value):
            print(f"警告: 摄像头不接受{name}设置 {value:g}")


def frame_quality(frame) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    clipped_percent = float((value >= 250).mean() * 100)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return clipped_percent, sharpness


def main() -> None:
    parser = argparse.ArgumentParser(description="采集目标检测训练图片")
    parser.add_argument("--tag", default="img", help="文件名前缀，便于区分场景/类别")
    parser.add_argument("--source", default="0", help="摄像头编号、视频路径，或 csi")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fixed-camera-fps", action="store_true", help="关闭自动曝光动态降帧")
    parser.add_argument("--auto-exposure", type=float, help="0.25/1=手动，0.75/3=自动")
    parser.add_argument("--exposure", type=float, help="手动曝光值")
    parser.add_argument("--gain", type=float, help="手动增益值")
    parser.add_argument("--backlight-compensation", type=float, help="背光补偿值")
    parser.add_argument("--gamma", type=float, help="Gamma值，用于调整中间调亮度")
    parser.add_argument("--quality-metrics", action="store_true", help="记录高光剪切率和清晰度")
    parser.add_argument(
        "--output-dir", type=Path, default=RAW_DIR,
        help="图片保存目录；训练批次和独立验证批次应使用不同目录",
    )
    parser.add_argument("--interval", type=float, default=0.0, help=">0 时按该秒数自动连拍")
    parser.add_argument("--no-display", action="store_true", help="无图形界面（SSH）时使用")
    parser.add_argument("--count", type=int, default=0, help="拍够这么多张自动退出，0 = 不限")
    args = parser.parse_args()

    headless = args.no_display or not os.environ.get("DISPLAY")
    if headless and args.interval <= 0:
        raise SystemExit("无窗口模式下没法按键拍照，请加 --interval（例如 --interval 0.5）")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.source, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"无法打开摄像头: {args.source}")
    if args.source != "csi":
        configure_uvc_camera(cap, args.source, args)

    actual = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"实际分辨率: {actual[0]}x{actual[1]}  摄像头报告帧率: {cap.get(cv2.CAP_PROP_FPS):.0f}")

    saved = 0
    last_auto = 0.0
    manifest_path = out_dir / "capture_manifest.csv"
    if headless:
        print(f"无窗口模式，每 {args.interval}s 自动拍一张"
              f"{f'，共 {args.count} 张' if args.count else ''}，Ctrl+C 停止")
    else:
        print("[s] 保存当前帧    [q] 退出")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取失败，退出")
                break

            now = time.time()
            auto_shot = args.interval > 0 and now - last_auto >= args.interval

            if headless:
                key = 0xFF
            else:
                preview = frame.copy()
                cv2.putText(
                    preview, f"saved: {saved}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
                cv2.imshow("capture", preview)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s") or auto_shot:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = out_dir / f"{args.tag}_{stamp}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                last_auto = now
                quality_text = ""
                if args.quality_metrics:
                    clipped_percent, sharpness = frame_quality(frame)
                    write_header = not manifest_path.exists()
                    with manifest_path.open("a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        if write_header:
                            writer.writerow([
                                "filename", "clipped_percent", "sharpness", "width", "height"
                            ])
                        writer.writerow([
                            path.name, f"{clipped_percent:.2f}", f"{sharpness:.1f}",
                            frame.shape[1], frame.shape[0],
                        ])
                    quality_text = (
                        f"  clip={clipped_percent:.2f}% sharp={sharpness:.1f}"
                    )
                print(f"[{saved}] {path.name}{quality_text}", flush=True)
                if args.count and saved >= args.count:
                    break
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"共保存 {saved} 张，目录: {out_dir}")


if __name__ == "__main__":
    main()
