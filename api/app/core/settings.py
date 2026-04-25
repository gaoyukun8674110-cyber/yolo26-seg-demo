from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str = "yolo26-mvtec-seg-demo"
    project_root: Path = Path(__file__).resolve().parents[3]


def get_settings() -> Settings:
    return Settings()
