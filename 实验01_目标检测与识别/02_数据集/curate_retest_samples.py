#!/usr/bin/env python3
"""将补测灰瓶正样本和耳机盒负样本整理为独立 YOLO 数据集。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
BOTTLE_CLASS_ID = 3

# xyxy 归一化框由旧实拍模型生成候选，并经逐图视觉核对。
SAMPLES = [
    ("微信图片_20260831184825_1288_7.jpg", "gray_thermos_01.jpg", (0.3170, 0.0085, 0.7660, 0.7663)),
    ("微信图片_20260831184830_1289_7.jpg", "gray_thermos_02.jpg", (0.2925, 0.0437, 0.7126, 0.7587)),
    ("微信图片_20260831184836_1290_7.jpg", "gray_thermos_03.jpg", (0.2618, 0.0233, 0.6695, 0.7695)),
    ("微信图片_20260831184843_1291_7.jpg", "gray_thermos_04.jpg", (0.2712, 0.0383, 0.6834, 0.7985)),
    ("微信图片_20260831184859_1292_7.jpg", "gray_thermos_05.jpg", (0.1947, 0.0000, 0.5896, 0.7000)),
    ("微信图片_20260831184907_1293_7.jpg", "gray_thermos_06.jpg", (0.2624, 0.0000, 0.6269, 0.6635)),
    ("微信图片_20260831184912_1294_7.jpg", "gray_thermos_07.jpg", (0.5611, 0.0120, 1.0000, 0.7088)),
    ("微信图片_20260831184922_1295_7.jpg", "gray_thermos_08.jpg", (0.3625, 0.1248, 0.7865, 0.7190)),
    ("微信图片_20260831184928_1296_7.jpg", "gray_thermos_09.jpg", (0.3098, 0.0344, 0.7348, 0.7938)),
    ("微信图片_20260831184937_1297_7.jpg", "gray_thermos_10.jpg", (0.5361, 0.0819, 0.9870, 0.6812)),
    ("微信图片_20260831184944_1298_7.jpg", "gray_thermos_11.jpg", (0.4797, 0.0186, 0.9867, 0.7937)),
    ("微信图片_20260831184952_1299_7.jpg", "gray_thermos_12.jpg", (0.3656, 0.0298, 0.6821, 0.5874)),
    ("微信图片_20260831184957_1300_7.jpg", "headphones_case_negative_01.jpg", None),
    ("微信图片_20260831185005_1301_7.jpg", "headphones_case_negative_02.jpg", None),
    ("微信图片_20260831185016_1302_7.jpg", "headphones_case_negative_03.jpg", None),
    ("微信图片_20260831185022_1303_7.jpg", "headphones_case_negative_04.jpg", None),
    ("微信图片_20260831185028_1304_7.jpg", "headphones_case_negative_05.jpg", None),
]


def parse_args():
    parser = argparse.ArgumentParser(description="整理补测灰瓶与耳机盒样本")
    parser.add_argument("--input", type=Path, default=ROOT.parents[1] / "补测")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "recollection" / "retest_20260831_yolo",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(output, overwrite):
    if output.exists():
        if not overwrite:
            raise SystemExit(f"输出目录已存在: {output}，使用 --overwrite 覆盖")
        shutil.rmtree(output)
    for relative in ("images/train", "labels/train", "previews"):
        (output / relative).mkdir(parents=True)


def yolo_line(box):
    left, top, right, bottom = box
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise SystemExit(f"非法归一化框: {box}")
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    width = right - left
    height = bottom - top
    return f"{BOTTLE_CLASS_ID} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"


def write_oriented_copy(source, destination):
    with Image.open(source) as image:
        orientation = image.getexif().get(274)
        fixed = ImageOps.exif_transpose(image).convert("RGB")
        fixed.save(destination, quality=95, subsampling=0)
    return orientation, fixed.size


def draw_preview(image_path, box, destination):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"无法解码整理后的图片: {image_path}")
    height, width = image.shape[:2]
    if box is not None:
        left, top, right, bottom = box
        point1 = (round(left * width), round(top * height))
        point2 = (round(right * width), round(bottom * height))
        thickness = max(4, round(min(width, height) / 350))
        cv2.rectangle(image, point1, point2, (20, 180, 20), thickness)
        cv2.putText(
            image, "bottle", (point1[0], max(35, point1[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 180, 20), thickness, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image, "NEGATIVE", (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
            1.8, (20, 20, 220), 4, cv2.LINE_AA,
        )
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise SystemExit(f"无法生成预览: {destination}")
    encoded.tofile(destination)


def make_contact_sheet(previews, destination):
    tile_width, tile_height = 360, 500
    columns, rows = 4, 5
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, path in enumerate(previews):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        scale = min(tile_width / image.shape[1], (tile_height - 40) / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        left, top = column * tile_width, row * tile_height
        x = left + (tile_width - resized.shape[1]) // 2
        y = top + 40 + (tile_height - 40 - resized.shape[0]) // 2
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(
            sheet, path.stem, (left + 8, top + 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (20, 20, 20), 1, cv2.LINE_AA,
        )
    success, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise SystemExit("无法生成联系表")
    encoded.tofile(destination)


def main():
    args = parse_args()
    source_root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"补测目录不存在: {source_root}")
    expected_names = {record[0] for record in SAMPLES}
    actual_names = {path.name for path in source_root.glob("*.jpg")}
    if actual_names != expected_names:
        raise SystemExit(
            f"补测图片清单发生变化，新增={sorted(actual_names - expected_names)}，"
            f"缺少={sorted(expected_names - actual_names)}"
        )

    prepare_output(output, args.overwrite)
    manifest = {"source": str(source_root), "samples": []}
    previews = []
    for source_name, output_name, box in SAMPLES:
        source = source_root / source_name
        image_destination = output / "images" / "train" / output_name
        label_destination = output / "labels" / "train" / Path(output_name).with_suffix(".txt")
        preview_destination = output / "previews" / output_name
        orientation, dimensions = write_oriented_copy(source, image_destination)
        label_destination.write_text(yolo_line(box) if box else "", encoding="utf-8")
        draw_preview(image_destination, box, preview_destination)
        previews.append(preview_destination)
        manifest["samples"].append({
            "source_filename": source_name,
            "output_filename": output_name,
            "usage": "bottle_positive" if box else "headphones_case_negative",
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(image_destination),
            "source_exif_orientation": orientation,
            "output_dimensions": list(dimensions),
            "label": yolo_line(box).strip() if box else "",
        })

    make_contact_sheet(previews, output / "contact_sheet.jpg")
    (output / "dataset.yaml").write_text(
        "train: images/train\n\nnames:\n"
        "  0: cup\n  1: mouse\n  2: keyboard\n  3: bottle\n",
        encoding="utf-8",
    )
    (output / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "images": len(SAMPLES),
        "bottle_positives": sum(box is not None for _, _, box in SAMPLES),
        "headphones_case_negatives": sum(box is None for _, _, box in SAMPLES),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()