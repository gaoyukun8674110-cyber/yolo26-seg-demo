import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "metrics" / "per_category.json"


def categories_from_manifest(data_root: Path, split: str) -> list[str]:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    return sorted({sample["category"] for sample in manifest["samples"] if sample["split"] == split})


def image_list_for_category(data_root: Path, category: str, split: str) -> list[Path]:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    return [
        data_root / "images" / split / sample["output_image_name"]
        for sample in manifest["samples"]
        if sample["split"] == split and sample["category"] == category
    ]


def metric_value(metrics: Any, *paths: str) -> float | None:
    for path in paths:
        current = metrics
        found = True
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                found = False
                break
        if found:
            try:
                return round(float(current), 6)
            except (TypeError, ValueError):
                continue
    return None


def extract_mask_metrics(results: Any) -> dict[str, float | None]:
    return {
        "precision_mask": metric_value(results, "seg.mp", "results_dict.metrics/precision(M)", "results_dict.metrics/precision(B)"),
        "recall_mask": metric_value(results, "seg.mr", "results_dict.metrics/recall(M)", "results_dict.metrics/recall(B)"),
        "map50_mask": metric_value(results, "seg.map50", "results_dict.metrics/mAP50(M)", "results_dict.metrics/mAP50(B)"),
        "map50_95_mask": metric_value(results, "seg.map", "results_dict.metrics/mAP50-95(M)", "results_dict.metrics/mAP50-95(B)"),
    }


def write_category_data_yaml(data_root: Path, category: str, split: str, work_dir: Path) -> Path:
    images = image_list_for_category(data_root, category, split)
    if not images:
        raise ValueError(f"No {split} images found for category {category}")

    list_path = work_dir / f"{category}_{split}.txt"
    list_path.write_text("\n".join(str(path.resolve()) for path in images) + "\n", encoding="utf-8")
    source_yaml = yaml.safe_load((data_root / "data.yaml").read_text(encoding="utf-8"))
    data_yaml = work_dir / f"data_{category}.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(data_root.resolve()),
                "train": str(list_path.resolve()),
                "val": str(list_path.resolve()),
                "test": str(list_path.resolve()),
                "names": source_yaml["names"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def evaluate_per_category(
    weight: Path,
    data_root: Path,
    split: str = "val",
    output_json: Path = DEFAULT_OUTPUT,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO

    model = model_factory(str(weight))
    per_category = {}
    with tempfile.TemporaryDirectory(prefix="yolo26-per-category-") as tmp:
        work_dir = Path(tmp)
        for category in categories_from_manifest(data_root, split):
            data_yaml = write_category_data_yaml(data_root, category, split, work_dir)
            results = model.val(data=str(data_yaml), split=split)
            per_category[category] = extract_mask_metrics(results)

    overall_results = model.val(data=str(data_root / "data.yaml"), split=split)
    payload = {
        "weight": str(weight),
        "split": split,
        "per_category": per_category,
        "overall": extract_mask_metrics(overall_results),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO segmentation weight separately for each MVTec category.")
    parser.add_argument("--weight", type=Path, required=True, help="YOLO segmentation weight to evaluate.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_balanced"), help="Processed dataset root.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    args = parser.parse_args()

    payload = evaluate_per_category(args.weight, args.data_root, args.split, args.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
