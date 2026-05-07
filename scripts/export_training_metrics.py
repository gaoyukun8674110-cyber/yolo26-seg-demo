import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MASK_METRIC_COLUMNS = {
    "precision_mask": "metrics/precision(M)",
    "recall_mask": "metrics/recall(M)",
    "map50_mask": "metrics/mAP50(M)",
    "map50_95_mask": "metrics/mAP50-95(M)",
}


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = float(text)
    if math.isnan(parsed):
        return None
    return parsed


def _parse_epoch(value: str | None) -> int:
    if value is None:
        raise ValueError("results.csv row is missing the epoch column")
    return int(float(value.strip()))


def parse_results_csv(results_csv: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with results_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{results_csv} does not contain a CSV header")
        for row in reader:
            rows.append({key.strip(): (value or "").strip() for key, value in row.items() if key is not None})

    if not rows:
        raise ValueError(f"{results_csv} does not contain any training rows")

    ranked_rows = []
    for row in rows:
        map50_mask = _parse_float(row.get(MASK_METRIC_COLUMNS["map50_mask"]))
        if map50_mask is not None:
            ranked_rows.append((map50_mask, row))

    if not ranked_rows:
        raise ValueError(f"{results_csv} does not contain usable mask mAP50 metrics")

    _, best_row = max(ranked_rows, key=lambda item: item[0])
    metrics: dict[str, Any] = {
        "training_epochs": len(rows),
        "best_epoch": _parse_epoch(best_row.get("epoch")),
        "best_selection_metric": "metrics/mAP50(M)",
        "metrics_source": results_csv.as_posix(),
    }
    for output_key, csv_key in MASK_METRIC_COLUMNS.items():
        value = _parse_float(best_row.get(csv_key))
        metrics[output_key] = round(value, 6) if value is not None else None
    return metrics


def export_training_metrics(
    results_csv: Path,
    summary_json: Path,
    model_path: Path,
) -> dict[str, Any]:
    summary = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

    payload = {
        **summary,
        "model_status": "trained-evaluated",
        "model_path": model_path.as_posix(),
        **parse_results_csv(results_csv),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO segmentation training metrics to artifacts/metrics/summary.json.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("runs/segment/yolo26s_seg_mvtec6_defect_v1/results.csv"),
        help="Path to the Ultralytics results.csv file.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("artifacts/metrics/summary.json"),
        help="Path to the metrics summary JSON to update.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("runs/segment/yolo26s_seg_mvtec6_defect_v1/weights/best.pt"),
        help="Path to the trained model weight represented by these metrics.",
    )
    args = parser.parse_args()

    payload = export_training_metrics(
        results_csv=args.results_csv,
        summary_json=args.summary_json,
        model_path=args.model_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
