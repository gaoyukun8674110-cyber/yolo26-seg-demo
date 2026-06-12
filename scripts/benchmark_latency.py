import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_WEIGHT = PROJECT_ROOT / "runs" / "segment" / "yolo26s_seg_mvtec6_defect_v1" / "weights" / "best.pt"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "test"
DEFAULT_SUMMARY = PROJECT_ROOT / "artifacts" / "metrics" / "summary.json"


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile for an empty sequence")
    if percentile_value < 0 or percentile_value > 100:
        raise ValueError("percentile must be in [0, 100]")

    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile_value / 100) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def select_images(image_dir: Path, samples: int, warmup: int, seed: int) -> list[Path]:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    required = samples + warmup
    rng = random.Random(seed)
    if required <= len(image_paths):
        return rng.sample(image_paths, required)
    return [rng.choice(image_paths) for _ in range(required)]


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_latency_benchmark(
    weight: Path,
    image_dir: Path,
    samples: int = 50,
    warmup: int = 5,
    seed: int = 0,
    predictor_factory: Callable[[Path, Path], object] | None = None,
) -> dict[str, float | int | str]:
    if samples <= 0:
        raise ValueError("--samples must be greater than 0")
    if warmup < 0:
        raise ValueError("--warmup cannot be negative")

    selected_images = select_images(image_dir=image_dir, samples=samples, warmup=warmup, seed=seed)
    if predictor_factory is None:
        from api.app.services.inference import UltralyticsSegmentationPredictor

        predictor_factory = UltralyticsSegmentationPredictor

    with TemporaryDirectory(prefix="yolo26-latency-") as generated:
        predictor = predictor_factory(weight, Path(generated))
        for image_path in selected_images[:warmup]:
            predictor.predict(image_path.read_bytes(), image_path.name)

        latencies = [
            float(predictor.predict(image_path.read_bytes(), image_path.name).latency_ms)
            for image_path in selected_images[warmup:]
        ]

    return {
        "latency_ms": round(median(latencies), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "latency_mean_ms": round(mean(latencies), 2),
        "latency_samples": len(latencies),
        "latency_device": detect_device(),
    }


def update_summary(summary_json: Path, metrics: dict[str, float | int | str]) -> dict[str, object]:
    summary: dict[str, object] = {}
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

    payload = {**summary, **metrics}
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real YOLO26 segmentation inference latency and update summary.json.")
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT, help="Path to the trained YOLO segmentation weight.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory of held-out images to sample.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY, help="Metrics summary JSON to update.")
    parser.add_argument("--samples", type=int, default=50, help="Timed prediction sample count.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup prediction count, excluded from timings.")
    parser.add_argument("--seed", type=int, default=0, help="Random sampling seed.")
    args = parser.parse_args()

    metrics = run_latency_benchmark(
        weight=args.weight,
        image_dir=args.image_dir,
        samples=args.samples,
        warmup=args.warmup,
        seed=args.seed,
    )
    payload = update_summary(args.summary_json, metrics)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
