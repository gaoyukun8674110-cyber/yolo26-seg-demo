from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml


Point = tuple[int, int]


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
