#!/usr/bin/env python3
"""为自动筛选后的 Open Images 正样本生成人工复核包。"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["cup", "mouse", "keyboard", "bottle"]
CLASS_COLORS = {
    0: (45, 180, 45),
    1: (220, 120, 30),
    2: (30, 150, 235),
    3: (190, 50, 190),
}


def parse_args():
    root = Path(__file__).resolve().parent
    dataset = root / "external_openimages_balanced_100_yolo"
    parser = argparse.ArgumentParser(
        description="Prepare a review package for automatically selected positives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=dataset)
    parser.add_argument("--output", type=Path, default=dataset / "待人工复核")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot decode image: {path}")
    return image


def write_jpeg(path: Path, image, quality=90):
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise SystemExit(f"Cannot encode preview: {path}")
    encoded.tofile(path)


def prepare_output(output: Path, overwrite: bool):
    if output.exists():
        if any(output.iterdir()) and not overwrite:
            raise SystemExit(f"Output is not empty: {output}. Use --overwrite.")
        if overwrite:
            shutil.rmtree(output)
    for name in ("原图", "标签", "带框预览"):
        for class_name in CLASS_NAMES:
            (output / name / class_name).mkdir(parents=True, exist_ok=True)
    (output / "联系表").mkdir(parents=True, exist_ok=True)


def load_labels(path: Path, width: int, height: int):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise SystemExit(f"Invalid YOLO label in {path}: {line}")
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        if class_id not in range(len(CLASS_NAMES)):
            raise SystemExit(f"Invalid class ID in {path}: {class_id}")
        if not all(0 <= value <= 1 for value in (center_x, center_y, box_width, box_height)):
            raise SystemExit(f"Invalid normalized coordinates in {path}: {line}")
        if box_width <= 0 or box_height <= 0:
            raise SystemExit(f"Non-positive box in {path}: {line}")
        labels.append({
            "class_id": class_id,
            "box": (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ),
        })
    if not labels:
        raise SystemExit(f"Selected positive has an empty label: {path}")
    return labels


def draw_labels(image, labels):
    annotated = image.copy()
    height, width = annotated.shape[:2]
    thickness = max(2, round(min(width, height) / 350))
    font_scale = max(0.55, min(width, height) / 1000)
    for label in labels:
        class_id = label["class_id"]
        x1, y1, x2, y2 = [int(round(value)) for value in label["box"]]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        color = CLASS_COLORS[class_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        text = CLASS_NAMES[class_id]
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            annotated,
            (x1, text_top),
            (min(width - 1, x1 + text_width + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 4, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def fit_image(image, width: int, height: int):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    offset_x = (width - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    canvas[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
    return canvas


def make_contact_sheets(records, output: Path):
    tile_width = 320
    tile_height = 260
    header_height = 42
    columns = 5
    rows = 5
    page_size = columns * rows
    for class_name, class_records in records.items():
        for start in range(0, len(class_records), page_size):
            page = class_records[start:start + page_size]
            sheet = np.full(
                (rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8
            )
            for index, record in enumerate(page):
                row, column = divmod(index, columns)
                left = column * tile_width
                top = row * tile_height
                preview = read_image(record["preview_path"])
                fitted = fit_image(preview, tile_width, tile_height - header_height)
                sheet[
                    top + header_height:top + tile_height,
                    left:left + tile_width,
                ] = fitted
                label = f"{start + index + 1:03d} {record['filename']}"
                cv2.putText(
                    sheet,
                    label[:43],
                    (left + 6, top + 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    sheet,
                    (left, top),
                    (left + tile_width - 1, top + tile_height - 1),
                    (155, 155, 155),
                    1,
                )
            page_number = start // page_size + 1
            write_jpeg(output / "联系表" / f"{class_name}_{page_number:02d}.jpg", sheet, 86)


def write_instructions(output: Path, counts):
    count_text = "，".join(f"{name} {counts[name]} 张" for name in CLASS_NAMES)
    (output / "复核说明.md").write_text(
        "# 正样本人工复核\n\n"
        f"本目录共 {sum(counts.values())} 张待复核图片：{count_text}。"
        "这些图片通过了自动质量筛选，但尚无逐文件人工确认状态。\n\n"
        "## 复核方法\n\n"
        "1. 先看 `联系表/`，每类 4 页，每页 25 张。\n"
        "2. 有疑问时打开 `带框预览/<类别>/` 和 `原图/<类别>/`。\n"
        "3. 在 `复核清单.csv` 的 `复核状态` 列把 `待复核` 改为 `保留`、`删除` 或 `修标签`。\n"
        "4. 在 `问题说明` 列记录漏标、错框、非实物、严重模糊等原因。\n\n"
        "请不要直接修改上一级 `images/train` 与 `labels/train`；本目录是独立副本。\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == dataset:
        raise SystemExit("Review output must not replace the selected dataset.")
    selection_path = dataset / "selection.json"
    if not selection_path.is_file():
        raise SystemExit(f"Missing selection manifest: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected", [])
    if not selected:
        raise SystemExit("Selection manifest has no selected records.")

    prepare_output(output, args.overwrite)
    rows = []
    contact_records = defaultdict(list)
    counts = Counter()
    for record in selected:
        filename = record["filename"]
        class_name = record["requested_class"]
        if class_name not in CLASS_NAMES:
            raise SystemExit(f"Unexpected requested class: {class_name}")
        image_source = dataset / "images" / "train" / filename
        label_source = dataset / "labels" / "train" / f"{Path(filename).stem}.txt"
        if not image_source.is_file() or not label_source.is_file():
            raise SystemExit(f"Missing selected image or label: {filename}")
        actual_hash = sha256_file(image_source)
        if actual_hash != record["sha256"]:
            raise SystemExit(f"Selected image hash mismatch: {filename}")

        image = read_image(image_source)
        height, width = image.shape[:2]
        labels = load_labels(label_source, width, height)
        image_destination = output / "原图" / class_name / filename
        label_destination = output / "标签" / class_name / label_source.name
        preview_destination = output / "带框预览" / class_name / filename
        shutil.copy2(image_source, image_destination)
        shutil.copy2(label_source, label_destination)
        write_jpeg(preview_destination, draw_labels(image, labels))

        rows.append({
            "类别": class_name,
            "文件名": filename,
            "复核状态": "待复核",
            "问题说明": "",
            "候选批次": record["candidate_batch"],
            "SHA256": actual_hash,
            "标注框数": len(labels),
            "当前模型漏检框数": record["screening"]["missed_ground_truth"],
        })
        contact_records[class_name].append({
            "filename": filename,
            "preview_path": preview_destination,
        })
        counts[class_name] += 1

    for class_name in CLASS_NAMES:
        contact_records[class_name].sort(key=lambda item: item["filename"])
    make_contact_sheets(contact_records, output)
    fieldnames = list(rows[0])
    with (output / "复核清单.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (CLASS_NAMES.index(item["类别"]), item["文件名"])))
    write_instructions(output, counts)

    expected = selection.get("counts", {}).get("selected_by_requested_class", {})
    if dict(counts) != expected:
        raise SystemExit(f"Review counts do not match selection manifest: {dict(counts)} != {expected}")
    print(json.dumps({"output": str(output), "counts": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()