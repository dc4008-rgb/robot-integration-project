#!/usr/bin/env python3
"""整理用户实拍干扰物，并隔离训练集与回归集。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


SELECTED_MODEL_SHA256 = (
    "b8dc30ef0b3dc008cbae6ab7ad08182c4232fefbd802215912639ce88077a520"
)

SAMPLES = {
    "386380d9b765fe3dc6bea583f5263c89.jpg": {
        "sha256": "4e9664e494c9b3c511f49f9ffef3a8a79ce70e1776821996ce1a76636d2f90ed",
        "usage": "train_mixed",
        "distractor": "headphones_case",
        "label": "2 0.460938 0.128881 0.921875 0.257762\n",
        "baseline_predictions": {"cup": [0.902486, 0.508454]},
    },
    "0fed40a004b28dee1789075eda03672e.jpg": {
        "sha256": "cb5f6999a779fd592b7cd35177cf26173d78c5dedb5b38c7edcfcf7bae30d4ff",
        "usage": "train_mixed",
        "distractor": "headphones_case",
        "label": "2 0.482422 0.134446 0.964844 0.268893\n",
        "baseline_predictions": {"cup": [0.744778], "mouse": [0.736994]},
    },
    "14ef6600f9bbd441720033b555557780.jpg": {
        "sha256": "063edf5448af75f5ec44616903000e8114a6031c92606bee163c65ed868536f7",
        "usage": "train_mixed",
        "distractor": "headphones_case",
        "label": "2 0.414062 0.114821 0.828125 0.229642\n",
        "baseline_predictions": {"mouse": [0.727019, 0.308644]},
    },
    "b533a2678fd5da9ef27932be4bdbd908.jpg": {
        "sha256": "3b33dbe50c05b08446ff88a85ba3face287df4afc17db43f3030b13c4305b29d",
        "usage": "regression_mixed",
        "distractor": "headphones_case",
        "label": "2 0.456250 0.103516 0.912500 0.207031\n",
        "baseline_predictions": {"cup": [0.488842]},
    },
    "aee1b83da707e0f0a6c89c267ff762fb.jpg": {
        "sha256": "7c55116e4ea4c44f4e91f934aca35d2455afcbdc258efc957ef0ab69aab0f499",
        "usage": "train_pure",
        "distractor": "power_adapter",
        "label": "",
        "baseline_predictions": {"bottle": [0.645762]},
    },
    "457a62ea10bb2d7074a14d4c8207e425.jpg": {
        "sha256": "a8d16ad7cad7fddd5beb65f0c9bc95972452cdbc236a9c5fe8dec0c8ae1de9e8",
        "usage": "train_pure",
        "distractor": "power_adapter",
        "label": "",
        "baseline_predictions": {"bottle": [0.465204, 0.300139]},
    },
    "5f254dc66e5ff6f1b19f828b85c9ae19.jpg": {
        "sha256": "b232a83a6a32b2addd8d6decc587c1b2193761ee3c5ebebadea0a1dffbe398d9",
        "usage": "train_pure",
        "distractor": "power_adapter",
        "label": "",
        "baseline_predictions": {},
    },
    "9ba648aaa73f9baeeadb14687ffbbe81.jpg": {
        "sha256": "8a04077e96191ead6985d7b33d531fb7eea58c1d784c65f185adce33f5841536",
        "usage": "regression_pure",
        "distractor": "power_adapter",
        "label": "",
        "baseline_predictions": {"bottle": [0.549691]},
    },
    "79af9daa977de1f474a10f7accc20b1b.jpg": {
        "sha256": "8ad944d25700cf2f5d98aa2b12c6b4efeadb11dc21635acddb57789a58d497cd",
        "usage": "regression_pure",
        "distractor": "power_adapter",
        "label": "",
        "baseline_predictions": {"bottle": [0.331347]},
    },
}


def parse_args():
    dataset_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Curate user-captured distractors into train and regression sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=dataset_root.parents[1] / "实际负样本",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_root / "recollection" / "actual_negatives",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(output: Path, overwrite: bool):
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    for usage in {sample["usage"] for sample in SAMPLES.values()}:
        (output / usage).mkdir(parents=True)


def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    prepare_output(output, args.overwrite)

    records = []
    for filename, sample in SAMPLES.items():
        source_image = source / filename
        if not source_image.is_file():
            raise SystemExit(f"Missing source image: {source_image}")
        actual_hash = sha256_file(source_image)
        if actual_hash != sample["sha256"]:
            raise SystemExit(
                f"Source hash mismatch for {filename}: {actual_hash} != {sample['sha256']}"
            )

        destination = output / sample["usage"]
        output_image = destination / f"actual_{sample['distractor']}_{filename}"
        output_label = output_image.with_suffix(".txt")
        shutil.copy2(source_image, output_image)
        output_label.write_text(sample["label"], encoding="utf-8")
        if sha256_file(output_image) != actual_hash:
            raise SystemExit(f"Copied image hash mismatch: {output_image}")
        records.append({
            "source_filename": filename,
            "output_filename": output_image.name,
            **sample,
        })

    manifest = {
        "selection_version": 1,
        "source_directory": str(source),
        "screening_model_sha256": SELECTED_MODEL_SHA256,
        "confidence_threshold": 0.25,
        "visual_audit": {
            "headphones_case_images": (
                "contain a real keyboard and therefore use keyboard labels; "
                "they are not pure negatives"
            ),
            "power_adapter_images": (
                "contain none of cup, mouse, keyboard, or bottle and use empty labels"
            ),
            "minimum_pairwise_phash_distance": 16,
            "exact_duplicate_count": 0,
        },
        "records": records,
    }
    (output / "selection.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    counts = {
        usage: sum(record["usage"] == usage for record in records)
        for usage in sorted({record["usage"] for record in records})
    }
    print(json.dumps(counts, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()