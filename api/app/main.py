from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.app.api.routes import build_router
from api.app.core.settings import get_settings
from api.app.services.inference import InferenceService


def create_app(inference_service: InferenceService | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name)
    artifacts_dir = settings.project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=artifacts_dir), name="artifacts")
    app.include_router(
        build_router(
            inference_service or InferenceService(generated_dir=artifacts_dir / "generated")
        )
    )
    return app


app = create_app()
