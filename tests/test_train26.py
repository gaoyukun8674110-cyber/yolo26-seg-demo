import argparse
from pathlib import Path
from types import SimpleNamespace

import train26


class FakeYOLO:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[tuple[str, dict]] = []
        self.trainer = SimpleNamespace(save_dir=None)

    def train(self, **kwargs) -> None:
        self.calls.append(("train", kwargs))

    def export(self, **kwargs) -> None:
        self.calls.append(("export", kwargs))

    def val(self) -> None:
        self.calls.append(("val", {}))


def test_train26_passes_clean_start_training_arguments(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "segment" / "exp"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"pt")
    (run_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M)",
                "1,0.1,0.2,0.3,0.4",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(train26, "DEFAULT_SUMMARY", tmp_path / "summary.json")

    model = FakeYOLO("yolo26s-seg.pt")
    model.trainer.save_dir = run_dir
    args = argparse.Namespace(
        data=Path("data/processed_balanced/data.yaml"),
        model="yolo26s-seg.pt",
        epochs=150,
        imgsz=1024,
        batch=-1,
        name="exp",
        seed=7,
    )

    summary = train26.train(args, yolo_factory=lambda model_path: model)

    train_call = model.calls[0]
    assert train_call[0] == "train"
    assert train_call[1]["data"] == "data\\processed_balanced\\data.yaml" or train_call[1]["data"] == "data/processed_balanced/data.yaml"
    assert train_call[1]["cos_lr"] is True
    assert train_call[1]["seed"] == 7
    assert train_call[1]["deterministic"] is True
    assert train_call[1]["patience"] == 50
    assert ("export", {"format": "onnx"}) in model.calls
    assert ("val", {}) in model.calls
    assert summary is not None
    assert summary["model_path"] == (run_dir / "weights" / "best.pt").as_posix()
