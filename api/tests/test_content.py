from fastapi.testclient import TestClient

from api.app.main import create_app


def test_metrics_returns_summary_payload() -> None:
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "defect-segmentation"
    assert "latency_ms" in payload


def test_examples_returns_gallery_items() -> None:
    client = TestClient(create_app())

    response = client.get("/examples")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["items"], list)
    assert payload["items"][0]["category"] == "bottle"
