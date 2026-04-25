import base64

from fastapi.testclient import TestClient

from api.app.main import create_app
from api.app.services.inference import InferenceResult


class FakeInferenceService:
    def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
        return InferenceResult(
            has_defect=True,
            confidence=0.91,
            latency_ms=12.3,
            overlay_filename="demo-overlay.png",
        )


def test_predict_returns_overlay_summary() -> None:
    app = create_app(inference_service=FakeInferenceService())
    client = TestClient(app)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Yl6kAAAAASUVORK5CYII="
    )

    response = client.post(
        "/predict",
        files={"file": ("sample.png", png_bytes, "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "has_defect": True,
        "confidence": 0.91,
        "latency_ms": 12.3,
        "overlay_url": "/artifacts/generated/demo-overlay.png",
    }
