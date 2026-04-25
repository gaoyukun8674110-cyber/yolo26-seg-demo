from pathlib import Path


def summarize_split(images_dir: Path, labels_dir: Path) -> dict[str, int]:
    image_count = sum(1 for path in images_dir.iterdir() if path.is_file())
    label_paths = list(labels_dir.glob("*.txt"))
    empty_label_count = sum(1 for path in label_paths if path.read_text(encoding="utf-8").strip() == "")
    return {
        "images": image_count,
        "labels": len(label_paths),
        "empty_labels": empty_label_count,
    }
