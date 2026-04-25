import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str = "yolo26-mvtec-seg-demo"
    project_root: Path = Path(__file__).resolve().parents[3]
    model_path: Path | None = None


def get_settings() -> Settings:
    model_path = os.getenv("YOLO26_MODEL_PATH")
    return Settings(model_path=Path(model_path) if model_path else None)
