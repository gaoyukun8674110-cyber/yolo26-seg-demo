from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


Point = tuple[int, int]


@dataclass(frozen=True)
class SampleRecord:
    category: str
    defect_type: str
    split: str
    source_image_path: str
    source_mask_path: str | None
    output_image_name: str
    has_defect: bool


def _connected_components(mask: np.ndarray) -> list[list[Point]]:
    foreground = mask > 0
    visited = np.zeros_like(foreground, dtype=bool)
    height, width = foreground.shape
    components: list[list[Point]] = []

    for row in range(height):
        for col in range(width):
            if not foreground[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            component: list[Point] = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_col, current_row))
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        next_row = current_row + row_offset
                        next_col = current_col + col_offset
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if visited[next_row, next_col] or not foreground[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            components.append(component)

    return components


def _cross(origin: Point, first: Point, second: Point) -> int:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])


def _convex_hull(points: Iterable[Point]) -> list[Point]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    lower: list[Point] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[Point] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _normalize_point(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    return round(x / width, 6), round(y / height, 6)


def mask_to_yolo_rows(mask: np.ndarray) -> list[str]:
    if mask.ndim != 2:
        raise ValueError("mask_to_yolo_rows expects a single-channel mask")

    height, width = mask.shape
    rows: list[str] = []
    for component in _connected_components(mask):
        hull = _convex_hull(component)
        if len(hull) < 3:
            continue

        coordinates: list[str] = []
        for x, y in hull:
            norm_x, norm_y = _normalize_point(x, y, width, height)
            coordinates.extend([str(norm_x), str(norm_y)])
        rows.append("0 " + " ".join(coordinates))

    return rows


def write_data_yaml(output_dir: Path) -> None:
    payload = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }
    (output_dir / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _iter_png_chunks(payload: bytes) -> Iterable[tuple[bytes, bytes]]:
    cursor = 8
    while cursor < len(payload):
        chunk_length = struct.unpack(">I", payload[cursor : cursor + 4])[0]
        chunk_name = payload[cursor + 4 : cursor + 8]
        chunk_data = payload[cursor + 8 : cursor + 8 + chunk_length]
        yield chunk_name, chunk_data
        cursor += 12 + chunk_length


def _apply_png_filter(filter_type: int, scanline: bytes, previous: bytes, bytes_per_pixel: int) -> bytes:
    result = bytearray(scanline)

    if filter_type == 0:
        return bytes(result)

    for index in range(len(result)):
        left = result[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        up = previous[index] if previous else 0
        upper_left = previous[index - bytes_per_pixel] if previous and index >= bytes_per_pixel else 0

        if filter_type == 1:
            value = result[index] + left
        elif filter_type == 2:
            value = result[index] + up
        elif filter_type == 3:
            value = result[index] + ((left + up) // 2)
        elif filter_type == 4:
            predictor = left + up - upper_left
            pa = abs(predictor - left)
            pb = abs(predictor - up)
            pc = abs(predictor - upper_left)
            if pa <= pb and pa <= pc:
                value = result[index] + left
            elif pb <= pc:
                value = result[index] + up
            else:
                value = result[index] + upper_left
        else:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")

        result[index] = value & 0xFF

    return bytes(result)


def load_png_mask(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Unsupported image format for mask: {path}")

    width = height = None
    bit_depth = color_type = None
    compressed = bytearray()

    for chunk_name, chunk_data in _iter_png_chunks(payload):
        if chunk_name == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_name == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_name == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"PNG header missing for mask: {path}")
    if bit_depth != 8:
        raise ValueError(f"Only 8-bit PNG masks are supported: {path}")

    channels = {0: 1, 2: 3}.get(color_type)
    if channels is None:
        raise ValueError(f"Unsupported PNG color type {color_type} for mask: {path}")

    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    rows = []
    previous = b""
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        filtered = raw[cursor : cursor + stride]
        cursor += stride
        restored = _apply_png_filter(filter_type, filtered, previous, channels)
        previous = restored
        rows.append(np.frombuffer(restored, dtype=np.uint8))

    image = np.stack(rows).reshape(height, width, channels)
    if channels == 1:
        return image[:, :, 0]
    return image[:, :, 0]


def _split_names(total: int, train_ratio: float, val_ratio: float) -> list[str]:
    if total <= 0:
        return []
    if total == 1:
        return ["train"]
    if total == 2:
        return ["train", "test"]

    val_count = max(1, int(round(total * val_ratio)))
    test_count = max(1, int(round(total * (1 - train_ratio - val_ratio))))
    train_count = total - val_count - test_count

    while train_count < 1:
        if val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        train_count = total - val_count - test_count

    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def _safe_stem(category_root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(category_root).with_suffix("")
    return "__".join((category_root.name, *relative.parts)).replace(" ", "_")


def _collect_category_records(
    category_root: Path,
    train_ratio: float,
    val_ratio: float,
) -> list[SampleRecord]:
    grouped: dict[str, list[tuple[Path, Path | None, bool]]] = {"good": []}
    grouped["good"].extend((path, None, False) for path in sorted((category_root / "train" / "good").glob("*.png")))
    grouped["good"].extend((path, None, False) for path in sorted((category_root / "test" / "good").glob("*.png")))

    test_root = category_root / "test"
    if test_root.exists():
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name != "good"):
            entries: list[tuple[Path, Path | None, bool]] = []
            for image_path in sorted(defect_dir.glob("*.png")):
                mask_path = category_root / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                entries.append((image_path, mask_path, True))
            grouped[defect_dir.name] = entries

    records: list[SampleRecord] = []
    for defect_type, samples in grouped.items():
        split_names = _split_names(len(samples), train_ratio, val_ratio)
        for split_name, (image_path, mask_path, has_defect) in zip(split_names, samples, strict=False):
            records.append(
                SampleRecord(
                    category=category_root.name,
                    defect_type=defect_type,
                    split=split_name,
                    source_image_path=str(image_path),
                    source_mask_path=str(mask_path) if mask_path else None,
                    output_image_name=f"{_safe_stem(category_root, image_path)}.png",
                    has_defect=has_defect,
                )
            )
    return records


def _write_label(output_path: Path, rows: list[str]) -> None:
    output_path.write_text("\n".join(rows), encoding="utf-8")


def _build_manifest(records: list[SampleRecord]) -> dict[str, object]:
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1

    return {
        "summary": {
            "total_samples": len(records),
            "categories": sorted({record.category for record in records}),
            "split_counts": split_counts,
            "defect_samples": sum(1 for record in records if record.has_defect),
            "good_samples": sum(1 for record in records if not record.has_defect),
        },
        "samples": [asdict(record) for record in records],
    }


def convert_mvtec_dataset(
    source_root: Path,
    output_root: Path,
    categories: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, object]:
    records: list[SampleRecord] = []
    for category in categories:
        category_root = source_root / category
        if not category_root.exists():
            raise FileNotFoundError(f"Missing category directory: {category_root}")
        records.extend(_collect_category_records(category_root, train_ratio, val_ratio))

    for split_name in ("train", "val", "test"):
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for record in records:
        source_image_path = Path(record.source_image_path)
        target_image_path = output_root / "images" / record.split / record.output_image_name
        target_label_path = output_root / "labels" / record.split / f"{Path(record.output_image_name).stem}.txt"

        shutil.copy2(source_image_path, target_image_path)
        if record.has_defect:
            mask = load_png_mask(Path(record.source_mask_path or ""))
            label_rows = mask_to_yolo_rows(mask)
        else:
            label_rows = []
        _write_label(target_label_path, label_rows)

    write_data_yaml(output_root)
    manifest = _build_manifest(records)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MVTec AD categories into YOLO segmentation format.")
    parser.add_argument("--source-root", type=Path, default=Path(r"H:\YOLO-Train\MVTec AD"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "capsule", "metal_nut"],
        help="Category folders to convert from the MVTec root.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_mvtec_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        categories=args.categories,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
