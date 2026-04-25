from fastapi.testclient import TestClient

from api.app.main import create_app


def build_client() -> TestClient:
    return TestClient(create_app())
