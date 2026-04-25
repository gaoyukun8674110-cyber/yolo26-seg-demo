from pathlib import Path

from scripts.validate_dataset import summarize_split


def test_summarize_split_counts_images_and_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "sample.png").write_bytes(b"png")
    (labels_dir / "sample.txt").write_text("", encoding="utf-8")

    summary = summarize_split(images_dir, labels_dir)

    assert summary["images"] == 1
    assert summary["labels"] == 1
    assert summary["empty_labels"] == 1
