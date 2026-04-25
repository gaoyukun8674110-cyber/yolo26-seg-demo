import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True)
class InferenceResult:
    has_defect: bool
    confidence: float
    latency_ms: float
    overlay_filename: str


class PredictorBackend(Protocol):
    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult: ...


class InferenceService:
    def __init__(
        self,
        generated_dir: Path | None = None,
        predictor: PredictorBackend | None = None,
    ) -> None:
        self._generated_dir = generated_dir
        self._predictor = predictor

    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        if self._predictor is not None:
            return self._predictor.predict(image_bytes, filename)

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


class UltralyticsSegmentationPredictor:
    def __init__(self, model_path: Path, generated_dir: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Configured model path does not exist: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'ultralytics' package to enable real YOLO26 inference."
            ) from exc

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Install 'opencv-python-headless' and 'numpy' to enable real YOLO26 inference."
            ) from exc

        self._generated_dir = generated_dir
        self._cv2 = cv2
        self._model = YOLO(str(model_path))
        self._np = np

    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        image = self._cv2.imdecode(
            self._np.frombuffer(image_bytes, dtype=self._np.uint8),
            self._cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("Unable to decode uploaded image.")

        self._generated_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(image_bytes).hexdigest()[:12]
        stem = Path(filename).stem or "upload"
        overlay_name = f"{stem}-{digest}-overlay.png"
        overlay_path = self._generated_dir / overlay_name

        started_at = perf_counter()
        result = self._model.predict(source=image, save=False, verbose=False)[0]
        latency_ms = (perf_counter() - started_at) * 1000

        overlay = result.plot()
        self._cv2.imwrite(str(overlay_path), overlay)

        boxes = getattr(result, "boxes", None)
        confidences = getattr(boxes, "conf", None)
        confidence_values = confidences.tolist() if confidences is not None else []
        masks = getattr(result, "masks", None)
        mask_data = getattr(masks, "data", None)
        has_masks = mask_data is not None and len(mask_data) > 0
        has_boxes = boxes is not None and len(boxes) > 0

        return InferenceResult(
            has_defect=bool(has_masks or has_boxes),
            confidence=float(max(confidence_values)) if confidence_values else 0.0,
            latency_ms=round(latency_ms, 2),
            overlay_filename=overlay_name,
        )
