from pathlib import PurePosixPath
from typing import Any


def build_gallery_payload(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "category": row["category"],
                "status": row["status"],
                "image": "/" + str(PurePosixPath(row["overlay_path"])),
            }
        )
    return {"items": items}
