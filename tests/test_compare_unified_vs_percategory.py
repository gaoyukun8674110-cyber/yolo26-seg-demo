import json
from pathlib import Path
from types import SimpleNamespace

from scripts.compare_unified_vs_percategory import compare_unified_vs_percategory


class FakeParameter:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


class FakeInnerModel:
    def parameters(self):
        return [FakeParameter(10), FakeParameter(15)]


class FakeModel:
    model = FakeInnerModel()

    def __init__(self, weight: str) -> None:
        self.weight = weight

    def val(self, data: str, split: str) -> SimpleNamespace:
        value = 0.8 if "bottle" in data else 0.6
        return SimpleNamespace(seg=SimpleNamespace(mp=value, mr=value, map50=value, map=value / 2))


def make_compare_dataset(root: Path) -> None:
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


def fake_latency_runner(**_kwargs):
    return {"latency_ms": 12.34}


def test_compare_unified_vs_percategory_writes_json_and_markdown(tmp_path: Path) -> None:
    data_root = tmp_path / "processed"
    make_compare_dataset(data_root)
    per_category = tmp_path / "per_category.json"
    per_category.write_text(
        json.dumps(
            {
                "weight": "unified.pt",
                "split": "val",
                "per_category": {
                    "bottle": {"map50_mask": 0.5},
                    "cable": {"map50_mask": 0.4},
                },
                "overall": {"map50_mask": 0.45},
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "comparison.json"
    output_table = tmp_path / "comparison.md"

    payload = compare_unified_vs_percategory(
        unified_weight=tmp_path / "unified.pt",
        percat_root=tmp_path / "runs" / "segment",
        data_root=data_root,
        per_category_json=per_category,
        output_json=output_json,
        output_table=output_table,
        model_factory=FakeModel,
        latency_runner=fake_latency_runner,
    )

    assert sorted(payload["categories"]) == ["bottle", "cable"]
    assert payload["categories"]["bottle"]["unified_map50"] == 0.5
    assert payload["categories"]["bottle"]["per_category_map50"] == 0.8
    assert payload["categories"]["bottle"]["parameters"] == 25
    assert payload["categories"]["bottle"]["latency_ms"] == 12.34
    assert payload["overall"]["unified_map50"] == 0.45
    assert output_json.exists()
    assert "| Overall |" in output_table.read_text(encoding="utf-8")
