from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    has_defect: bool
    confidence: float
    latency_ms: float
    overlay_filename: str


class InferenceService:
    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        raise NotImplementedError("Attach a real YOLO26 predictor after the API contract is stable.")
