from fastapi.testclient import TestClient

from api.app.main import create_app


def test_health_returns_ok_payload() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "yolo26-mvtec-seg-demo"}
