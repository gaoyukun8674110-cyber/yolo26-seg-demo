import json
from pathlib import Path

from scripts.make_per_category_datasets import make_per_category_datasets


def write_sample(root: Path, category: str, split: str, kind: str, index: int, has_defect: bool) -> dict:
    image_name = f"{category}__{split}__{kind}__{index:03d}.png"
    image = root / "images" / split / image_name
    label = root / "labels" / split / Path(image_name).with_suffix(".txt").name
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    label.write_text("0 0.1 0.1 0.2 0.2\n" if has_defect else "", encoding="utf-8")
    return {
        "category": category,
        "defect_type": kind,
        "split": split,
        "source_image_path": str(image),
        "source_mask_path": None,
        "output_image_name": image_name,
        "has_defect": has_defect,
    }


def test_make_per_category_datasets_filters_and_rebalances(tmp_path: Path) -> None:
    source = tmp_path / "processed"
    samples = []
    for category in ["bottle", "cable"]:
        for index in range(4):
            samples.append(write_sample(source, category, "train", "good", index, False))
        samples.append(write_sample(source, category, "train", "defect", 0, True))
        samples.append(write_sample(source, category, "val", "good", 0, False))
        samples.append(write_sample(source, category, "test", "defect", 0, True))
    (source / "manifest.json").write_text(json.dumps({"summary": {}, "samples": samples}), encoding="utf-8")
    (source / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n", encoding="utf-8")

    summaries = make_per_category_datasets(source, tmp_path / "per_category", good_ratio=1.0, seed=0)

    assert sorted(summaries) == ["bottle", "cable"]
    bottle_manifest = json.loads((tmp_path / "per_category" / "bottle" / "manifest.json").read_text(encoding="utf-8"))
    assert {sample["category"] for sample in bottle_manifest["samples"]} == {"bottle"}
    assert bottle_manifest["summary"]["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert (tmp_path / "per_category" / "bottle" / "data_bottle.yaml").exists()
