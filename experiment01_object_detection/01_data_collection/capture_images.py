"""Camera capture script for collecting training images on a Jetson or local computer.

Usage:
    python capture_images.py --tag cup                    # Press s to save one image; q to quit
    python capture_images.py --tag mixed --interval 0.5   # Automatically save one image every 0.5 seconds
    python capture_images.py --source csi                 # Jetson CSI camera
    python capture_images.py --tag mouse --output-dir data/train

When operating remotely over SSH without a graphical interface, use headless mode:
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

RAW_DIR = Path(__file__).resolve().parent.parent / "02_\u6570\u636e\u96c6" / "raw"


def csi_pipeline(width: int, height: int, fps: int) -> str:
    """Return the GStreamer pipeline for a Jetson CSI camera such as the IMX219."""
    return (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
    )


def open_camera(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if source == "csi":
        return cv2.VideoCapture(csi_pipeline(width, height, fps), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    # UVC cameras default to uncompressed YUYV, which saturates bandwidth at 720p and yields only 5-10 FPS; switch to MJPG before setting the resolution.
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
                print(f"Warning: camera rejected {name}={value:g}: {message}")
            else:
                print(f"Camera {name}={value:g}")
        return

    settings = [
        (cv2.CAP_PROP_AUTO_EXPOSURE, args.auto_exposure, "auto-exposure"),
        (cv2.CAP_PROP_EXPOSURE, args.exposure, "exposure"),
        (cv2.CAP_PROP_GAIN, args.gain, "gain"),
        (cv2.CAP_PROP_BACKLIGHT, args.backlight_compensation, "backlight compensation"),
        (cv2.CAP_PROP_GAMMA, args.gamma, "Gamma"),
    ]
    for property_id, value, name in settings:
        if value is not None and not cap.set(property_id, value):
            print(f"Warning: camera rejected {name} setting {value:g}")


def frame_quality(frame) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    clipped_percent = float((value >= 250).mean() * 100)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return clipped_percent, sharpness


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture training images for object detection")
    parser.add_argument("--tag", default="img", help="Filename prefix used to distinguish scenes/classes")
    parser.add_argument("--source", default="0", help="Camera index, video path, or csi")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fixed-camera-fps", action="store_true", help="Disable dynamic frame-rate reduction by auto-exposure")
    parser.add_argument("--auto-exposure", type=float, help="0.25/1=manual, 0.75/3=auto")
    parser.add_argument("--exposure", type=float, help="Manual exposure value")
    parser.add_argument("--gain", type=float, help="Manual gain value")
    parser.add_argument("--backlight-compensation", type=float, help="Backlight compensation value")
    parser.add_argument("--gamma", type=float, help="Gamma value for adjusting midtone brightness")
    parser.add_argument("--quality-metrics", action="store_true", help="Record highlight clipping percentage and sharpness")
    parser.add_argument(
        "--output-dir", type=Path, default=RAW_DIR,
        help="Image output directory; use separate directories for training and independent validation batches",
    )
    parser.add_argument("--interval", type=float, default=0.0, help="Automatically capture at this interval in seconds when >0")
    parser.add_argument("--no-display", action="store_true", help="Use without a graphical interface (SSH)")
    parser.add_argument("--count", type=int, default=0, help="Exit after capturing this many images; 0 = unlimited")
    args = parser.parse_args()

    headless = args.no_display or not os.environ.get("DISPLAY")
    if headless and args.interval <= 0:
        raise SystemExit("Keyboard capture is unavailable in headless mode; add --interval (for example, --interval 0.5)")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.source, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open camera: {args.source}")
    if args.source != "csi":
        configure_uvc_camera(cap, args.source, args)

    actual = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"Actual resolution: {actual[0]}x{actual[1]}  Camera-reported frame rate: {cap.get(cv2.CAP_PROP_FPS):.0f}")

    saved = 0
    last_auto = 0.0
    manifest_path = out_dir / "capture_manifest.csv"
    if headless:
        print(f"Headless mode: automatically capture one image every {args.interval}s"
              f"{f', {args.count} images total' if args.count else ''}; press Ctrl+C to stop")
    else:
        print("[s] Save current frame    [q] Quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame; exiting")
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
        print("\nInterrupted")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"Saved {saved} images in total; directory: {out_dir}")


if __name__ == "__main__":
    main()
