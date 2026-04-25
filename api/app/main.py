from fastapi import FastAPI

from api.app.api.routes import build_router
from api.app.core.settings import get_settings
from api.app.services.inference import InferenceService


def create_app(inference_service: InferenceService | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name)
    app.include_router(build_router(inference_service or InferenceService()))
    return app


app = create_app()
