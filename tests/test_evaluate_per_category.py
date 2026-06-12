import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_per_category import evaluate_per_category


class FakeModel:
    def __init__(self, _weight: str) -> None:
        self.val_paths: list[str] = []

    def val(self, data: str, split: str) -> SimpleNamespace:
        self.val_paths.append(data)
        text = Path(data).read_text(encoding="utf-8")
        if "bottle" in text:
            value = 0.7
        elif "cable" in text:
            value = 0.4
        else:
            value = 0.5
        return SimpleNamespace(seg=SimpleNamespace(mp=value, mr=value + 0.01, map50=value + 0.02, map=value + 0.03))


def make_eval_dataset(root: Path) -> None:
    samples = []
    for category in ["bottle", "cable"]:
        image = root / "images" / "val" / f"{category}__val__good__000.png"
        label = root / "labels" / "val" / f"{category}__val__good__000.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        label.write_text("", encoding="utf-8")
        samples.append(
            {
                "category": category,
                "defect_type": "good",
                "split": "val",
                "source_image_path": str(image),
                "source_mask_path": None,
                "output_image_name": image.name,
                "has_defect": False,
            }
        )
    (root / "manifest.json").write_text(json.dumps({"summary": {}, "samples": samples}), encoding="utf-8")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n", encoding="utf-8")


def test_evaluate_per_category_writes_expected_json(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    output = tmp_path / "per_category.json"
    make_eval_dataset(data_root)

    payload = evaluate_per_category(
        weight=tmp_path / "best.pt",
        data_root=data_root,
        split="val",
        output_json=output,
        model_factory=FakeModel,
    )

    assert sorted(payload["per_category"]) == ["bottle", "cable"]
    assert payload["per_category"]["bottle"]["map50_mask"] == 0.72
    assert payload["per_category"]["cable"]["map50_mask"] == 0.42
    assert payload["overall"]["map50_mask"] == 0.52
    assert json.loads(output.read_text(encoding="utf-8")) == payload
