"""USB camera diagnostic: identify devices and formats, then measure actual MJPG and YUYV capture rates.

FPS is an acceptance metric, but the UVC camera output format is often the bottleneck
(uncompressed YUYV commonly achieves only 5-10 FPS at 720p).
Run this script first to establish the camera limit before optimizing model speed.

Usage:
    python3 check_camera.py                 # Scan devices automatically and test at 1280x720
    python3 check_camera.py --index 0 --width 640 --height 480
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2

FORMATS = {"MJPG": "MJPG (compressed, recommended)", "YUYV": "YUYV (uncompressed, slow)"}


def list_devices() -> list[int]:
    nodes = sorted(Path("/dev").glob("video*"))
    if nodes:
        print("Detected device nodes:", ", ".join(p.name for p in nodes))
        return [int(p.name.replace("video", "")) for p in nodes]
    print("No /dev/video* devices found (normal on macOS; trying index 0 directly)")
    return [0]


def show_v4l2_formats(index: int) -> None:
    if not shutil.which("v4l2-ctl"):
        print("Tip: install v4l-utils to list every format supported by the camera")
        print("      sudo apt install v4l-utils\n")
        return
    out = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{index}", "--list-formats-ext"],
        capture_output=True, text=True,
    ).stdout
    print(out or "(no output from v4l2-ctl)")


def measure(index: int, fourcc: str, width: int, height: int, n: int = 60) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  {fourcc}: unable to open device {index}")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    got = int(cap.get(cv2.CAP_PROP_FOURCC))
    got_str = "".join(chr((got >> 8 * i) & 0xFF) for i in range(4))
    actual = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"

    for _ in range(10):
        cap.read()

    t0 = time.time()
    ok_count = sum(1 for _ in range(n) if cap.read()[0])
    fps = ok_count / (time.time() - t0)
    cap.release()

    warn = ""
    if got_str != fourcc:
        warn = f"  Warning: effective format is {got_str}; the camera may not support {fourcc}"
    print(f"  {FORMATS[fourcc]:22s} {actual:>10s}  Measured {fps:5.1f} FPS{warn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="USB camera diagnostic")
    parser.add_argument("--index", type=int, default=None, help="Scan all devices when omitted")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    indices = [args.index] if args.index is not None else list_devices()

    for idx in indices:
        print(f"\n===== /dev/video{idx} =====")
        show_v4l2_formats(idx)
        for fourcc in FORMATS:
            measure(idx, fourcc, args.width, args.height)

    print("\nInterpretation:")
    print("  MJPG much faster than YUYV -> Normal; the capture script already defaults to MJPG")
    print("  MJPG also below 15 FPS     -> Retest at 640x480; if still slow, camera/USB bandwidth is the limit")
    print("  Both formats are slow      -> Use a USB 3.0 port and ensure the camera is not connected through a dock/HUB")


if __name__ == "__main__":
    main()
