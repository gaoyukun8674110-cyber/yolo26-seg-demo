import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_latency import run_latency_benchmark
from scripts.evaluate_per_category import (
    DEFAULT_OUTPUT as DEFAULT_PER_CATEGORY,
    evaluate_per_category,
    extract_mask_metrics,
    write_category_data_yaml,
)


DEFAULT_COMPARISON_JSON = PROJECT_ROOT / "artifacts" / "metrics" / "comparison.json"
DEFAULT_TABLE = PROJECT_ROOT / "docs" / "comparison_table.md"


def model_parameter_count(model: Any) -> int | None:
    inner = getattr(model, "model", model)
    if hasattr(inner, "parameters"):
        try:
            return int(sum(parameter.numel() for parameter in inner.parameters()))
        except TypeError:
            return None
    return getattr(inner, "param_count", None)


def category_weight(percat_root: Path, category: str) -> Path:
    return percat_root / f"percat_{category}_v1" / "weights" / "best.pt"


def mean_number(values: list[float | int | None]) -> float | None:
    real_values = [float(value) for value in values if value is not None]
    if not real_values:
        return None
    return round(sum(real_values) / len(real_values), 6)


def write_markdown_table(output_path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "| Category | Unified mAP50 | Per-category mAP50 | Params | Latency ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, row in comparison["categories"].items():
        lines.append(
            "| {category} | {unified:.6f} | {percat:.6f} | {params} | {latency:.2f} |".format(
                category=category,
                unified=row["unified_map50"] or 0,
                percat=row["per_category_map50"] or 0,
                params=row["parameters"] if row["parameters"] is not None else "",
                latency=row["latency_ms"] or 0,
            )
        )
    overall = comparison["overall"]
    lines.append(
        "| Overall | {unified:.6f} | {percat:.6f} | {params} | {latency:.2f} |".format(
            unified=overall["unified_map50"] or 0,
            percat=overall["per_category_map50"] or 0,
            params=overall["parameters"] if overall["parameters"] is not None else "",
            latency=overall["latency_ms"] or 0,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_unified_vs_percategory(
    unified_weight: Path,
    percat_root: Path,
    data_root: Path,
    per_category_json: Path = DEFAULT_PER_CATEGORY,
    output_json: Path = DEFAULT_COMPARISON_JSON,
    output_table: Path = DEFAULT_TABLE,
    split: str = "val",
    latency_samples: int = 10,
    model_factory: Any | None = None,
    latency_runner=run_latency_benchmark,
) -> dict[str, Any]:
    if not per_category_json.exists():
        evaluate_per_category(unified_weight, data_root, split, per_category_json, model_factory=model_factory)
    unified_metrics = json.loads(per_category_json.read_text(encoding="utf-8"))

    if model_factory is None:
        from ultralytics import YOLO

        model_factory = YOLO

    rows = {}
    for category, unified_row in unified_metrics["per_category"].items():
        weight = category_weight(percat_root, category)
        model = model_factory(str(weight))
        with tempfile_directory() as work_dir:
            data_yaml = write_category_data_yaml(data_root, category, split, work_dir)
            metrics = extract_mask_metrics(model.val(data=str(data_yaml), split=split))

        latency = latency_runner(
            weight=weight,
            image_dir=data_root / "images" / split,
            samples=latency_samples,
            warmup=1,
            seed=0,
        )
        rows[category] = {
            "unified_map50": unified_row["map50_mask"],
            "per_category_map50": metrics["map50_mask"],
            "parameters": model_parameter_count(model),
            "latency_ms": latency["latency_ms"],
            "weight": str(weight),
        }

    payload = {
        "unified_weight": str(unified_weight),
        "split": split,
        "categories": rows,
        "overall": {
            "unified_map50": mean_number([row["unified_map50"] for row in rows.values()]),
            "per_category_map50": mean_number([row["per_category_map50"] for row in rows.values()]),
            "parameters": mean_number([row["parameters"] for row in rows.values()]),
            "latency_ms": mean_number([row["latency_ms"] for row in rows.values()]),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_table(output_table, payload)
    return payload


class tempfile_directory:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="yolo26-compare-")
        return Path(self._tmp.__enter__())

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.__exit__(exc_type, exc, tb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one unified defect model with per-category YOLO baselines.")
    parser.add_argument("--unified-weight", type=Path, required=True, help="Unified model weight.")
    parser.add_argument("--percat-root", type=Path, default=Path("runs/segment"), help="Root containing percat_{category}_v1 runs.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_balanced"), help="Balanced unified dataset root.")
    parser.add_argument("--per-category-json", type=Path, default=DEFAULT_PER_CATEGORY, help="Unified per-category metrics JSON.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_COMPARISON_JSON, help="Comparison JSON output.")
    parser.add_argument("--output-table", type=Path, default=DEFAULT_TABLE, help="Markdown table output.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split to compare.")
    parser.add_argument("--latency-samples", type=int, default=10, help="Timed samples per per-category model.")
    args = parser.parse_args()

    payload = compare_unified_vs_percategory(
        unified_weight=args.unified_weight,
        percat_root=args.percat_root,
        data_root=args.data_root,
        per_category_json=args.per_category_json,
        output_json=args.output_json,
        output_table=args.output_table,
        split=args.split,
        latency_samples=args.latency_samples,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
