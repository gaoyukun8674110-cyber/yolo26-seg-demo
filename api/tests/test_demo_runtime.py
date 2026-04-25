import base64
from pathlib import Path

from fastapi.testclient import TestClient

import api.app.api.routes as routes_module
import api.app.main as main_module
from api.app.core.settings import Settings
from api.app.services.inference import InferenceResult


def _sample_png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Yl6kAAAAASUVORK5CYII="
    )


def test_default_predict_writes_generated_overlay(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    client = TestClient(main_module.create_app())

    response = client.post(
        "/predict",
        files={"file": ("sample.png", _sample_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["overlay_url"].startswith("/artifacts/generated/")
    generated_dir = tmp_path / "artifacts" / "generated"
    assert generated_dir.exists()
    assert any(generated_dir.glob("*.png"))


def test_app_serves_artifact_files(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(project_root=tmp_path)
    artifact_path = tmp_path / "artifacts" / "examples" / "demo.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"ok": true}', encoding="utf-8")
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    client = TestClient(main_module.create_app())

    response = client.get("/artifacts/examples/demo.json")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_create_app_uses_real_predictor_when_model_path_is_configured(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "weights.pt"
    model_path.write_bytes(b"weights")
    settings = Settings(project_root=tmp_path, model_path=model_path)
    calls: dict[str, Path] = {}

    class FakePredictor:
        def __init__(self, configured_model_path: Path, generated_dir: Path) -> None:
            calls["model_path"] = configured_model_path
            calls["generated_dir"] = generated_dir

        def predict(self, image_bytes: bytes, filename: str) -> InferenceResult:
            output_name = "real-overlay.png"
            calls["generated_dir"].mkdir(parents=True, exist_ok=True)
            (calls["generated_dir"] / output_name).write_bytes(image_bytes)
            return InferenceResult(
                has_defect=True,
                confidence=0.84,
                latency_ms=7.5,
                overlay_filename=output_name,
            )

    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "UltralyticsSegmentationPredictor", FakePredictor)

    client = TestClient(main_module.create_app())
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _sample_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert calls["model_path"] == model_path
    assert response.json() == {
        "has_defect": True,
        "confidence": 0.84,
        "latency_ms": 7.5,
        "overlay_url": "/artifacts/generated/real-overlay.png",
    }
