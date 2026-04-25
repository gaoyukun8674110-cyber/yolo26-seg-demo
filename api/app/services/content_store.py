import json
from pathlib import Path
from typing import Any


class ContentStore:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def read_metrics(self) -> dict[str, Any]:
        metrics_path = self._project_root / "artifacts" / "metrics" / "summary.json"
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def read_examples(self) -> dict[str, Any]:
        examples_path = self._project_root / "artifacts" / "examples" / "gallery.json"
        return json.loads(examples_path.read_text(encoding="utf-8"))
