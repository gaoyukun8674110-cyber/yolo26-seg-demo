import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rebalance_dataset import SPLITS, copy_sample, summarize_samples


def select_category_samples(samples: list[dict], category: str, good_ratio: float, seed: int) -> list[dict]:
    category_samples = [sample for sample in samples if sample["category"] == category]
    train_good = [sample for sample in category_samples if sample["split"] == "train" and not sample["has_defect"]]
    train_defect = [sample for sample in category_samples if sample["split"] == "train" and sample["has_defect"]]
    keep_good_count = min(len(train_good), round(len(train_defect) * good_ratio))
    keep_good_names = {
        sample["output_image_name"]
        for sample in random.Random(seed).sample(train_good, keep_good_count)
    }
    return [
        sample
        for sample in category_samples
        if sample["split"] != "train" or sample["has_defect"] or sample["output_image_name"] in keep_good_names
    ]


def write_category_yaml(source: Path, output: Path, category: str) -> None:
    source_yaml = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    payload = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": source_yaml["names"],
    }
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    (output / f"data_{category}.yaml").write_text(text, encoding="utf-8")
    (output / "data.yaml").write_text(text, encoding="utf-8")


def make_per_category_datasets(
    source: Path,
    output_root: Path,
    good_ratio: float = 1.5,
    seed: int = 0,
) -> dict[str, dict]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted({sample["category"] for sample in manifest["samples"]})
    results = {}

    for category in categories:
        output = output_root / category
        if output.exists():
            shutil.rmtree(output)
        for split in SPLITS:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)

        selected = select_category_samples(manifest["samples"], category, good_ratio, seed)
        for sample in selected:
            copy_sample(source, output, sample)

        category_manifest = {"summary": summarize_samples(selected), "samples": selected}
        (output / "manifest.json").write_text(
            json.dumps(category_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_category_yaml(source, output, category)
        results[category] = category_manifest["summary"]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one balanced YOLO dataset per MVTec category.")
    parser.add_argument("--source", type=Path, default=Path("data/processed"), help="Source processed dataset root.")
    parser.add_argument("--output-root", type=Path, default=Path("data/per_category"), help="Output root for category datasets.")
    parser.add_argument("--good-ratio", type=float, default=1.5, help="Target train good:defect ratio per category.")
    parser.add_argument("--seed", type=int, default=0, help="Random sampling seed.")
    args = parser.parse_args()

    payload = make_per_category_datasets(args.source, args.output_root, args.good_ratio, args.seed)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
