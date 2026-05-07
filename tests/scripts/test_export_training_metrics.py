import json
from pathlib import Path

from scripts.export_training_metrics import export_training_metrics, parse_results_csv


def test_parse_results_csv_selects_best_epoch_by_mask_map50(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "\n".join(
            [
                "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)",
                "1,0.10,0.20,0.30,0.05",
                "2,0.30,0.25,0.45,0.08",
                "3,0.40,0.30,0.41,0.09",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_results_csv(results_path)

    assert metrics["training_epochs"] == 3
    assert metrics["best_epoch"] == 2
    assert metrics["precision_mask"] == 0.3
    assert metrics["recall_mask"] == 0.25
    assert metrics["map50_mask"] == 0.45
    assert metrics["map50_95_mask"] == 0.08


def test_export_training_metrics_preserves_dataset_summary_and_writes_model_metrics(tmp_path: Path) -> None:
    results_path = tmp_path / "runs" / "segment" / "exp" / "results.csv"
    results_path.parent.mkdir(parents=True)
    results_path.write_text(
        "\n".join(
            [
                "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)",
                "1,0.10,0.20,0.30,0.05",
                "2,0.30,0.25,0.45,0.08",
            ]
        ),
        encoding="utf-8",
    )
    existing_summary = tmp_path / "artifacts" / "metrics" / "summary.json"
    existing_summary.parent.mkdir(parents=True)
    existing_summary.write_text(
        json.dumps(
            {
                "task": "defect-segmentation",
                "categories": ["bottle"],
                "total_samples": 2,
                "latency_ms": 0,
            }
        ),
        encoding="utf-8",
    )

    payload = export_training_metrics(
        results_csv=results_path,
        summary_json=existing_summary,
        model_path=Path("runs/segment/exp/weights/best.pt"),
    )

    written = json.loads(existing_summary.read_text(encoding="utf-8"))
    assert payload == written
    assert written["task"] == "defect-segmentation"
    assert written["model_status"] == "trained-evaluated"
    assert written["model_path"] == "runs/segment/exp/weights/best.pt"
    assert written["training_epochs"] == 2
    assert written["best_epoch"] == 2
    assert written["precision_mask"] == 0.3
    assert written["recall_mask"] == 0.25
    assert written["map50_mask"] == 0.45
    assert written["map50_95_mask"] == 0.08
