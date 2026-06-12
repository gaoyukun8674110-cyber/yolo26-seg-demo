import json
from pathlib import Path

from scripts.rebalance_dataset import rebalance_dataset


def write_sample(root: Path, split: str, image_name: str, has_defect: bool) -> dict:
    image_path = root / "images" / split / image_name
    label_path = root / "labels" / split / Path(image_name).with_suffix(".txt").name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")
    label_path.write_text("0 0.1 0.1 0.2 0.2\n" if has_defect else "", encoding="utf-8")
    defect_type = "defect" if has_defect else "good"
    return {
        "category": "bottle",
        "defect_type": defect_type,
        "split": split,
        "source_image_path": str(image_path),
        "source_mask_path": None,
        "output_image_name": image_name,
        "has_defect": has_defect,
    }


def make_dataset(root: Path) -> None:
    samples = []
    for index in range(6):
        samples.append(write_sample(root, "train", f"bottle__train__good__{index:03d}.png", False))
    for index in range(2):
        samples.append(write_sample(root, "train", f"bottle__train__defect__{index:03d}.png", True))
    samples.append(write_sample(root, "val", "bottle__val__good__000.png", False))
    samples.append(write_sample(root, "test", "bottle__test__defect__000.png", True))
    (root / "manifest.json").write_text(json.dumps({"summary": {}, "samples": samples}), encoding="utf-8")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n", encoding="utf-8")


def test_rebalance_dataset_downsamples_train_good_only(tmp_path: Path) -> None:
    source = tmp_path / "processed"
    output = tmp_path / "processed_balanced"
    make_dataset(source)

    manifest = rebalance_dataset(source, output, good_ratio=1.5, seed=0)

    train_samples = [sample for sample in manifest["samples"] if sample["split"] == "train"]
    train_good = [sample for sample in train_samples if not sample["has_defect"]]
    train_defect = [sample for sample in train_samples if sample["has_defect"]]
    assert len(train_good) == 3
    assert len(train_defect) == 2
    assert manifest["summary"]["split_counts"]["val"] == 1
    assert manifest["summary"]["split_counts"]["test"] == 1
    assert (output / "images" / "val" / "bottle__val__good__000.png").exists()
    assert (output / "labels" / "test" / "bottle__test__defect__000.txt").exists()
    assert str(output.resolve()) in (output / "data.yaml").read_text(encoding="utf-8")
