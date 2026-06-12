import json
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark_latency import percentile, run_latency_benchmark, update_summary


class FakePredictor:
    def __init__(self, _weight: Path, _generated_dir: Path) -> None:
        self._latencies = iter([1.0, 2.0, 10.0, 20.0, 30.0, 40.0])

    def predict(self, _image_bytes: bytes, _filename: str) -> SimpleNamespace:
        return SimpleNamespace(latency_ms=next(self._latencies))


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([10, 20, 30, 40], 95) == 38.5


def test_benchmark_records_timed_samples_after_warmup(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(6):
        (image_dir / f"sample-{index}.png").write_bytes(b"png")

    metrics = run_latency_benchmark(
        weight=tmp_path / "best.pt",
        image_dir=image_dir,
        samples=4,
        warmup=2,
        seed=0,
        predictor_factory=FakePredictor,
    )

    assert metrics["latency_ms"] == 25
    assert metrics["latency_p95_ms"] == 38.5
    assert metrics["latency_samples"] == 4
    assert metrics["latency_device"] in {"cpu", "cuda"}


def test_update_summary_preserves_existing_fields(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"task": "defect-segmentation", "latency_ms": 0}), encoding="utf-8")

    payload = update_summary(summary, {"latency_ms": 25.12, "latency_p95_ms": 40.0, "latency_samples": 4})

    assert payload["task"] == "defect-segmentation"
    assert payload["latency_ms"] == 25.12
    assert json.loads(summary.read_text(encoding="utf-8")) == payload
