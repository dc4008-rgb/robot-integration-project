"""USB 摄像头体检：确认设备号、支持的格式，并实测 MJPG 与 YUYV 的真实采集帧率。

FPS 是验收指标，而 UVC 摄像头的输出格式往往才是瓶颈（YUYV 未压缩，720p 常只有 5~10 FPS）。
先跑这个脚本确认摄像头上限，再去优化模型速度。

用法:
    python3 check_camera.py                 # 自动扫描设备并测 1280x720
    python3 check_camera.py --index 0 --width 640 --height 480
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import cv2

FORMATS = {"MJPG": "MJPG（压缩，推荐）", "YUYV": "YUYV（未压缩，慢）"}


def list_devices() -> list[int]:
    nodes = sorted(Path("/dev").glob("video*"))
    if nodes:
        print("检测到设备节点:", ", ".join(p.name for p in nodes))
        return [int(p.name.replace("video", "")) for p in nodes]
    print("未找到 /dev/video*（macOS 上正常，直接试 index 0）")
    return [0]


def show_v4l2_formats(index: int) -> None:
    if not shutil.which("v4l2-ctl"):
        print("提示: 装上 v4l-utils 可以看到摄像头支持的全部格式")
        print("      sudo apt install v4l-utils\n")
        return
    out = subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{index}", "--list-formats-ext"],
        capture_output=True, text=True,
    ).stdout
    print(out or "(v4l2-ctl 无输出)")


def measure(index: int, fourcc: str, width: int, height: int, n: int = 60) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  {fourcc}: 无法打开设备 {index}")
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
        warn = f"  ⚠️ 实际生效的是 {got_str}，摄像头可能不支持 {fourcc}"
    print(f"  {FORMATS[fourcc]:22s} {actual:>10s}  实测 {fps:5.1f} FPS{warn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="USB 摄像头体检")
    parser.add_argument("--index", type=int, default=None, help="不指定则扫描全部设备")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    indices = [args.index] if args.index is not None else list_devices()

    for idx in indices:
        print(f"\n===== /dev/video{idx} =====")
        show_v4l2_formats(idx)
        for fourcc in FORMATS:
            measure(idx, fourcc, args.width, args.height)

    print("\n判读:")
    print("  MJPG 明显快于 YUYV  -> 正常，脚本已默认使用 MJPG")
    print("  MJPG 也低于 15 FPS  -> 降到 640x480 再测，仍慢则是摄像头/USB 带宽限制")
    print("  两者都很慢          -> 换 USB 3.0 口，或确认没插在扩展坞/HUB 上")


if __name__ == "__main__":
    main()
