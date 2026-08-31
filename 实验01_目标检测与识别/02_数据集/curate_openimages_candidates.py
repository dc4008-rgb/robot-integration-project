#!/usr/bin/env python3
"""Materialize the visually audited Open Images augmentation subset."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


EXPECTED_MODEL_SHA256 = (
    "b8dc30ef0b3dc008cbae6ab7ad08182c4232fefbd802215912639ce88077a520"
)

POSITIVE_BY_CLASS = {
    "cup": [
        "cup_90d8c0d1b5c64782.jpg",
        "cup_51748aa83946cad2.jpg",
        "cup_ce207316b0692329.jpg",
        "cup_80375f8a3a447d78.jpg",
    ],
    "mouse": [
        "mouse_53563ab5c237722a.jpg",
        "mouse_14bc75c712747602.jpg",
        "mouse_1c83d79add8d8797.jpg",
        "mouse_05a1c4450311751c.jpg",
        "mouse_0a538666793eb2c9.jpg",
    ],
    "keyboard": [
        "keyboard_643ab7cced3ea581.jpg",
        "keyboard_016cf78edc4a99af.jpg",
        "keyboard_044ecfc80793a365.jpg",
        "keyboard_92a5f7d360c55e18.jpg",
        "keyboard_acd46f19d9402180.jpg",
    ],
    "bottle": [
        "bottle_4b341cc067ef6805.jpg",
    ],
}

NEGATIVE_TRAIN_BY_CLASS = {
    "mobile_phone": [
        "mobile_phone_a8907dfe5129d96c.jpg",
        "mobile_phone_b160d7343a73e5af.jpg",
        "mobile_phone_02db97298559d7f4.jpg",
        "mobile_phone_4b00777f898051cc.jpg",
        "mobile_phone_379167d9b37e96b7.jpg",
        "mobile_phone_0f24d414f019ab84.jpg",
        "mobile_phone_ecf6de2e43a535c7.jpg",
        "mobile_phone_7b10596a3fcff2e6.jpg",
        "mobile_phone_25d902fdca9452ff.jpg",
        "mobile_phone_df5f753c74c1332a.jpg",
        "mobile_phone_697446b900357852.jpg",
        "mobile_phone_31228b0d51fb2886.jpg",
        "mobile_phone_583404f0291effa6.jpg",
        "mobile_phone_0c8ce872a79e42b3.jpg",
    ],
    "headphones": [
        "headphones_62e629f87a146dc7.jpg",
        "headphones_480d31422e410271.jpg",
        "headphones_989b1c379a6e8e33.jpg",
        "headphones_87a58045edeb303c.jpg",
        "headphones_e724795e201eda80.jpg",
        "headphones_b32b8b40995da7c2.jpg",
        "headphones_0fc8334c3b65dcf0.jpg",
        "headphones_ef3223724397d441.jpg",
        "headphones_923512f0fdb3920e.jpg",
        "headphones_8aa3b9f9074901b2.jpg",
    ],
    "power_plugs": [
        "power_plugs_d7fec8ce698343c9.jpg",
        "power_plugs_79ad72dcb7951d3b.jpg",
        "power_plugs_0dd42b6907f9d661.jpg",
        "power_plugs_de6dfa01d8f9da8d.jpg",
        "power_plugs_04234cffc3e51a9b.jpg",
        "power_plugs_fd557fbb42e8624a.jpg",
        "power_plugs_1f3282ecf59eb36e.jpg",
        "power_plugs_12f2971624fd374b.jpg",
        "power_plugs_3221bec581ae7178.jpg",
        "power_plugs_3cc7f2445d49a7f8.jpg",
        "power_plugs_9ba4a57a527dc78a.jpg",
        "power_plugs_78f7cc18dd1bb02f.jpg",
    ],
}

NEGATIVE_REGRESSION_BY_CLASS = {
    "mobile_phone": [
        "mobile_phone_a153fd2d19c6b0e7.jpg",
        "mobile_phone_6dbd48b4cf3f8220.jpg",
        "mobile_phone_b3433bc26084d558.jpg",
        "mobile_phone_fb2dfcbd8622a5f4.jpg",
        "mobile_phone_0da3109f36ec1e24.jpg",
    ],
    "headphones": [
        "headphones_422a23b200009cc1.jpg",
        "headphones_170156e8b4a8e47d.jpg",
        "headphones_73bb3c7b5a3bcdbf.jpg",
        "headphones_67799471cc3b8124.jpg",
        "headphones_216b175ba421c6d4.jpg",
    ],
    "power_plugs": [
        "power_plugs_c24e47e5ee6a3113.jpg",
        "power_plugs_b63586ed51bec115.jpg",
        "power_plugs_75f13fa9b550e2bc.jpg",
        "power_plugs_0982ea6be8d0ffaa.jpg",
        "power_plugs_2c558a2d13528bd2.jpg",
    ],
}

CRITICAL_VISUAL_EXCLUSIONS = {
    "mobile_phone_371845893907308e.jpg": "contains real cups",
    "mobile_phone_ea419ae8e84274dd.jpg": "contains a real bottle",
    "mobile_phone_c23fa0d4c4fa8abe.jpg": "contains real cups",
    "mobile_phone_ec56e1adcfeee297.jpg": "unrelated fermentation vessel",
    "headphones_8b6847e41fbd6703.jpg": "contains a real keyboard and mouse",
    "headphones_7d5a3c401b477a0e.jpg": "phone lens kit, not headphones",
    "headphones_13b0f7b58350e25f.jpg": "camera lens cap, not headphones",
    "headphones_80f3b521255e4976.jpg": "record-player stylus, not headphones",
    "power_plugs_4e34ba85ad937051.jpg": "synthetic advertising image",
    "power_plugs_ae5ee56dc9367f64.jpg": "payment card reader, not a power plug",
    "power_plugs_4111bad480f6f704.jpg": "kettle is the dominant object",
    "power_plugs_4e19ca779ce6b5a0.jpg": "synthetic face-like outlet image",
    "power_plugs_904c12f6fd7d22b1.jpg": "graphic outlet image, not a real scene",
    "mouse_dfd67385a0c0c127.jpg": "ground-truth box truncates the mouse",
    "mouse_4b402fd33c54f5b5.jpg": "poor image quality and possible missed input device",
    "keyboard_d554ec118748764b.jpg": "visible mouse is not labeled",
    "keyboard_8afb02bfe09695ae.jpg": "visible mouse is not labeled",
    "keyboard_372d28ae187eb22c.jpg": "keyboard box omits several key rows",
    "keyboard_40664a4a2a1944d1.jpg": "possible unlabeled bottle-shaped object",
    "cup_343f05d64f91c29c.jpg": "ground-truth box covers nearly the whole image",
    "cup_159a88aafb5d2bd5.jpg": "tiny cup in a person-centric scene",
}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Materialize the visually audited Open Images subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "external_openimages_bidirectional_candidates",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(groups):
    return [filename for filenames in groups.values() for filename in filenames]


def prepare_output(output: Path, overwrite: bool):
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    (output / "positive_yolo" / "images" / "train").mkdir(parents=True)
    (output / "positive_yolo" / "labels" / "train").mkdir(parents=True)
    (output / "negative_train").mkdir(parents=True)
    (output / "negative_regression").mkdir(parents=True)


def copy_checked(source: Path, destination: Path, expected_hash=None):
    if not source.is_file():
        raise SystemExit(f"Missing source file: {source}")
    actual_hash = sha256_file(source)
    if expected_hash is not None and actual_hash != expected_hash:
        raise SystemExit(
            f"Source hash mismatch for {source.name}: {actual_hash} != {expected_hash}"
        )
    shutil.copy2(source, destination)
    if sha256_file(destination) != actual_hash:
        raise SystemExit(f"Copied file hash mismatch: {destination}")
    return actual_hash


def build_record(filename, usage, group, source_records, screening_records):
    source_record = source_records.get(filename)
    screening_record = screening_records.get(filename)
    if source_record is None:
        raise SystemExit(f"Missing source manifest record: {filename}")
    if screening_record is None:
        raise SystemExit(f"Missing screening record: {filename}")
    return {
        "filename": filename,
        "usage": usage,
        "selection_group": group,
        "source": source_record,
        "screening": screening_record,
    }


def materialize_positive(
    candidates, output, filename, group, source_records, screening_records
):
    source_image = candidates / "positive" / "images" / "train" / filename
    source_label = (
        candidates / "positive" / "labels" / "train" /
        Path(filename).with_suffix(".txt").name
    )
    record = build_record(
        filename, "positive_train", group, source_records, screening_records
    )
    if record["source"].get("kind") != "positive":
        raise SystemExit(f"Source manifest does not mark positive: {filename}")
    if record["screening"].get("kind") != "positive":
        raise SystemExit(f"Screening report does not mark positive: {filename}")
    image_hash = copy_checked(
        source_image,
        output / "positive_yolo" / "images" / "train" / filename,
        record["source"].get("sha256"),
    )
    if not source_label.is_file() or not source_label.read_text().strip():
        raise SystemExit(f"Positive label is missing or empty: {source_label}")
    label_hash = copy_checked(
        source_label,
        output / "positive_yolo" / "labels" / "train" / source_label.name,
    )
    record["output_image_sha256"] = image_hash
    record["output_label_sha256"] = label_hash
    return record


def materialize_negative(
    candidates, output, filename, usage, group, source_records, screening_records
):
    source_image = candidates / "negative_train" / filename
    source_label = source_image.with_suffix(".txt")
    record = build_record(filename, usage, group, source_records, screening_records)
    if record["source"].get("kind") != "negative":
        raise SystemExit(f"Source manifest does not mark negative: {filename}")
    if record["screening"].get("kind") != "negative":
        raise SystemExit(f"Screening report does not mark negative: {filename}")
    if not source_label.is_file() or source_label.read_text().strip():
        raise SystemExit(f"Negative label must exist and be empty: {source_label}")
    destination = output / usage
    image_hash = copy_checked(
        source_image, destination / filename, record["source"].get("sha256")
    )
    label_hash = copy_checked(
        source_label, destination / source_label.name
    )
    record["output_image_sha256"] = image_hash
    record["output_label_sha256"] = label_hash
    return record


def write_dataset_yaml(output):
    (output / "positive_yolo" / "dataset.yaml").write_text(
        "train: images/train\n\n"
        "names:\n"
        "  0: cup\n"
        "  1: mouse\n"
        "  2: keyboard\n"
        "  3: bottle\n",
        encoding="utf-8",
    )


def count_positive_boxes(output):
    counts = Counter()
    label_dir = output / "positive_yolo" / "labels" / "train"
    for path in label_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise SystemExit(f"Invalid YOLO label in {path}: {line}")
            class_id = int(fields[0])
            coordinates = list(map(float, fields[1:]))
            if class_id not in range(4) or not all(0 <= value <= 1 for value in coordinates):
                raise SystemExit(f"Invalid YOLO values in {path}: {line}")
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise SystemExit(f"Non-positive YOLO box in {path}: {line}")
            counts[class_id] += 1
    return {str(class_id): counts[class_id] for class_id in range(4)}


def main():
    args = parse_args()
    candidates = args.candidates.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output else candidates / "selected"
    )
    source_path = candidates / "source.json"
    screening_path = candidates / "screening" / "screening.json"
    if not source_path.is_file() or not screening_path.is_file():
        raise SystemExit("Candidates must contain source.json and screening/screening.json.")

    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    screening_report = json.loads(screening_path.read_text(encoding="utf-8"))
    if screening_report.get("weights_sha256") != EXPECTED_MODEL_SHA256:
        raise SystemExit(
            "Screening report was not produced by the selected recollection model."
        )
    source_records = {
        record["filename"]: record for record in source_manifest["selected"]
    }
    screening_records = {
        record["filename"]: record for record in screening_report["records"]
    }

    positives = flatten(POSITIVE_BY_CLASS)
    negative_train = flatten(NEGATIVE_TRAIN_BY_CLASS)
    negative_regression = flatten(NEGATIVE_REGRESSION_BY_CLASS)
    all_selected = positives + negative_train + negative_regression
    if len(set(all_selected)) != len(all_selected):
        raise SystemExit("Selection lists overlap or contain duplicates.")

    prepare_output(output, args.overwrite)
    selected_records = []
    for group, filenames in POSITIVE_BY_CLASS.items():
        for filename in filenames:
            selected_records.append(materialize_positive(
                candidates, output, filename, group,
                source_records, screening_records,
            ))
    for group, filenames in NEGATIVE_TRAIN_BY_CLASS.items():
        for filename in filenames:
            selected_records.append(materialize_negative(
                candidates, output, filename, "negative_train", group,
                source_records, screening_records,
            ))
    for group, filenames in NEGATIVE_REGRESSION_BY_CLASS.items():
        for filename in filenames:
            selected_records.append(materialize_negative(
                candidates, output, filename, "negative_regression", group,
                source_records, screening_records,
            ))
    write_dataset_yaml(output)

    manifest = {
        "selection_version": 1,
        "candidate_source": str(candidates),
        "screening_model_sha256": EXPECTED_MODEL_SHA256,
        "screening_confidence_threshold": screening_report["confidence_threshold"],
        "selection_basis": [
            "Open Images metadata and project-target exclusion",
            "SHA-256 deduplication against the existing train and validation images",
            "selected-model difficulty screening",
            "visual review of all ten contact sheets and risky source images",
        ],
        "counts": {
            "positive_train_images": len(positives),
            "positive_train_by_requested_class": {
                key: len(value) for key, value in POSITIVE_BY_CLASS.items()
            },
            "positive_train_box_counts": count_positive_boxes(output),
            "negative_train_images": len(negative_train),
            "negative_train_by_distractor": {
                key: len(value) for key, value in NEGATIVE_TRAIN_BY_CLASS.items()
            },
            "negative_regression_images": len(negative_regression),
            "negative_regression_by_distractor": {
                key: len(value)
                for key, value in NEGATIVE_REGRESSION_BY_CLASS.items()
            },
        },
        "critical_visual_exclusions": CRITICAL_VISUAL_EXCLUSIONS,
        "selected": selected_records,
    }
    (output / "selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()