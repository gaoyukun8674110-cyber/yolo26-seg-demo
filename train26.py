import argparse
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

from scripts.export_training_metrics import export_training_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "processed_balanced" / "data.yaml"
DEFAULT_MODEL = "yolo26s-seg.pt"
DEFAULT_RUN_NAME = "yolo26s_seg_mvtec6_balanced_v1"
DEFAULT_SUMMARY = PROJECT_ROOT / "artifacts" / "metrics" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26-style MVTec defect segmentation. Requires a GPU runtime.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Dataset data.yaml path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base YOLO segmentation weight for a clean start.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epoch count.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Training batch size; -1 lets Ultralytics choose.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Ultralytics run name under runs/segment.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic training seed.")
    return parser.parse_args()


def resolve_run_dir(model: Any, run_name: str) -> Path:
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is not None:
        return Path(save_dir)
    return PROJECT_ROOT / "runs" / "segment" / run_name


def train(args: argparse.Namespace, yolo_factory: Any | None = None) -> dict | None:
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    model = yolo_factory(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device="cuda",
        batch=args.batch,
        workers=12,
        name=args.name,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        patience=50,
    )

    run_dir = resolve_run_dir(model, args.name)
    best_weight = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    summary = None
    if results_csv.exists():
        summary = export_training_metrics(results_csv=results_csv, summary_json=DEFAULT_SUMMARY, model_path=best_weight)

    model.export(format="onnx")
    model.val()
    return summary


def main():
    train(parse_args())


if __name__ == "__main__":
    freeze_support()
    main()
