import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml


SPLITS = ("train", "val", "test")


def load_manifest(source: Path) -> dict:
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def select_balanced_samples(samples: list[dict], good_ratio: float, seed: int) -> list[dict]:
    if good_ratio <= 0:
        raise ValueError("--good-ratio must be greater than 0")

    train_good = [sample for sample in samples if sample["split"] == "train" and not sample["has_defect"]]
    train_defect = [sample for sample in samples if sample["split"] == "train" and sample["has_defect"]]
    keep_good_count = min(len(train_good), round(len(train_defect) * good_ratio))
    keep_good_names = {
        sample["output_image_name"]
        for sample in random.Random(seed).sample(train_good, keep_good_count)
    }

    selected = []
    for sample in samples:
        if sample["split"] != "train" or sample["has_defect"] or sample["output_image_name"] in keep_good_names:
            selected.append(sample)
    return selected


def summarize_samples(samples: Iterable[dict]) -> dict:
    sample_list = list(samples)
    split_counts = Counter(sample["split"] for sample in sample_list)
    categories = sorted({sample["category"] for sample in sample_list})
    return {
        "total_samples": len(sample_list),
        "categories": categories,
        "split_counts": {split: split_counts.get(split, 0) for split in SPLITS},
        "defect_samples": sum(1 for sample in sample_list if sample["has_defect"]),
        "good_samples": sum(1 for sample in sample_list if not sample["has_defect"]),
    }


def copy_sample(source: Path, output: Path, sample: dict) -> None:
    split = sample["split"]
    image_name = sample["output_image_name"]
    label_name = Path(image_name).with_suffix(".txt").name
    source_image = source / "images" / split / image_name
    source_label = source / "labels" / split / label_name
    output_image = output / "images" / split / image_name
    output_label = output / "labels" / split / label_name

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, output_image)
    shutil.copy2(source_label, output_label)


def write_data_yaml(source: Path, output: Path, yaml_name: str = "data.yaml") -> None:
    source_yaml = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    payload = {
        **source_yaml,
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    (output / yaml_name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def rebalance_dataset(source: Path, output: Path, good_ratio: float = 1.5, seed: int = 0) -> dict:
    manifest = load_manifest(source)
    selected_samples = select_balanced_samples(manifest["samples"], good_ratio=good_ratio, seed=seed)

    if output.exists():
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for sample in selected_samples:
        copy_sample(source, output, sample)

    balanced_manifest = {"summary": summarize_samples(selected_samples), "samples": selected_samples}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(balanced_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_data_yaml(source, output)
    return balanced_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample train/good samples without changing val/test.")
    parser.add_argument("--source", type=Path, default=Path("data/processed"), help="Source processed dataset root.")
    parser.add_argument("--output", type=Path, default=Path("data/processed_balanced"), help="Output balanced dataset root.")
    parser.add_argument("--good-ratio", type=float, default=1.5, help="Target train good:defect ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Random sampling seed.")
    args = parser.parse_args()

    manifest = rebalance_dataset(args.source, args.output, args.good_ratio, args.seed)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
