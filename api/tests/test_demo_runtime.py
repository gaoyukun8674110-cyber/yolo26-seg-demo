import base64
from pathlib import Path

from fastapi.testclient import TestClient

import api.app.api.routes as routes_module
import api.app.main as main_module
from api.app.core.settings import Settings


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
