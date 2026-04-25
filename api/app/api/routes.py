from fastapi import APIRouter

from api.app.core.settings import get_settings
from api.app.services.content_store import ContentStore

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "service": settings.service_name}


@router.get("/metrics")
def metrics() -> dict:
    settings = get_settings()
    return ContentStore(settings.project_root).read_metrics()


@router.get("/examples")
def examples() -> dict:
    settings = get_settings()
    return ContentStore(settings.project_root).read_examples()
