import json
import struct
import zlib
from pathlib import Path

import numpy as np

from scripts.convert_mvtec_to_yolo_seg import convert_mvtec_dataset, mask_to_yolo_rows


def test_mask_to_yolo_rows_converts_a_single_rectangle() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 255

    rows = mask_to_yolo_rows(mask)

    assert len(rows) == 1
    assert rows[0].startswith("0 ")


def test_mask_to_yolo_rows_returns_empty_for_blank_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)

    rows = mask_to_yolo_rows(mask)

    assert rows == []


def _write_grayscale_png(path: Path, pixels: np.ndarray) -> None:
    height, width = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].astype(np.uint8).tobytes() for row in range(height))
    compressed = zlib.compress(raw)

    def chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)


def test_convert_mvtec_dataset_builds_processed_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "mvtec"
    category_root = source_root / "bottle"
    (category_root / "train" / "good").mkdir(parents=True)
    (category_root / "test" / "good").mkdir(parents=True)
    (category_root / "test" / "broken").mkdir(parents=True)
    (category_root / "ground_truth" / "broken").mkdir(parents=True)

    image_pixels = np.zeros((6, 6), dtype=np.uint8)
    image_pixels[1:5, 1:5] = 180
    _write_grayscale_png(category_root / "train" / "good" / "000.png", image_pixels)
    _write_grayscale_png(category_root / "test" / "good" / "001.png", image_pixels)
    _write_grayscale_png(category_root / "test" / "broken" / "002.png", image_pixels)

    mask_pixels = np.zeros((6, 6), dtype=np.uint8)
    mask_pixels[2:5, 2:5] = 255
    _write_grayscale_png(category_root / "ground_truth" / "broken" / "002_mask.png", mask_pixels)

    output_root = tmp_path / "processed"

    convert_mvtec_dataset(source_root, output_root, categories=["bottle"])

    image_paths = list((output_root / "images").rglob("*.png"))
    label_paths = list((output_root / "labels").rglob("*.txt"))
    manifest_path = output_root / "manifest.json"
    data_yaml_path = output_root / "data.yaml"

    assert len(image_paths) == 3
    assert len(label_paths) == 3
    assert manifest_path.exists()
    assert data_yaml_path.exists()

    defect_label = next(path for path in label_paths if "broken" in path.stem)
    good_label = next(path for path in label_paths if "good" in path.stem)

    assert defect_label.read_text(encoding="utf-8").startswith("0 ")
    assert good_label.read_text(encoding="utf-8") == ""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["total_samples"] == 3
    assert manifest["summary"]["categories"] == ["bottle"]


def test_convert_mvtec_dataset_can_rerun_on_existing_output(tmp_path: Path) -> None:
    source_root = tmp_path / "mvtec"
    category_root = source_root / "bottle"
    (category_root / "train" / "good").mkdir(parents=True)
    (category_root / "test" / "good").mkdir(parents=True)
    (category_root / "test" / "broken").mkdir(parents=True)
    (category_root / "ground_truth" / "broken").mkdir(parents=True)

    image_pixels = np.zeros((6, 6), dtype=np.uint8)
    image_pixels[1:5, 1:5] = 180
    train_image = category_root / "train" / "good" / "000.png"
    good_image = category_root / "test" / "good" / "001.png"
    defect_image = category_root / "test" / "broken" / "002.png"
    _write_grayscale_png(train_image, image_pixels)
    _write_grayscale_png(good_image, image_pixels)
    _write_grayscale_png(defect_image, image_pixels)
    train_image.chmod(0o444)
    good_image.chmod(0o444)
    defect_image.chmod(0o444)

    mask_pixels = np.zeros((6, 6), dtype=np.uint8)
    mask_pixels[2:5, 2:5] = 255
    _write_grayscale_png(category_root / "ground_truth" / "broken" / "002_mask.png", mask_pixels)

    output_root = tmp_path / "processed"

    convert_mvtec_dataset(source_root, output_root, categories=["bottle"])
    convert_mvtec_dataset(source_root, output_root, categories=["bottle"])

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["total_samples"] == 3
