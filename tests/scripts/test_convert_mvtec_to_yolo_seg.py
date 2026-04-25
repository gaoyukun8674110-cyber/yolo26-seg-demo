import numpy as np

from scripts.convert_mvtec_to_yolo_seg import mask_to_yolo_rows


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
