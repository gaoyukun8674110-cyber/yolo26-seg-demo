import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceResult:
    has_defect: bool
    confidence: float
    latency_ms: float
    overlay_filename: str


class InferenceService:
    def __init__(self, generated_dir: Path | None = None) -> None:
        self._generated_dir = generated_dir

    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        if self._generated_dir is None:
            raise NotImplementedError("Attach a real YOLO26 predictor after the API contract is stable.")

        self._generated_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix or ".png"
        digest = hashlib.sha1(image_bytes).hexdigest()[:12]
        stem = Path(filename).stem or "upload"
        output_name = f"{stem}-{digest}{suffix}"
        (self._generated_dir / output_name).write_bytes(image_bytes)

        return InferenceResult(
            has_defect=False,
            confidence=0.0,
            latency_ms=0.0,
            overlay_filename=output_name,
        )
