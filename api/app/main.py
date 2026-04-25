from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.app.api.routes import build_router
from api.app.core.settings import Settings, get_settings
from api.app.services.inference import InferenceService, UltralyticsSegmentationPredictor


def build_default_inference_service(settings: Settings, artifacts_dir) -> InferenceService:
    generated_dir = artifacts_dir / "generated"
    if settings.model_path is None:
        return InferenceService(generated_dir=generated_dir)

    return InferenceService(
        generated_dir=generated_dir,
        predictor=UltralyticsSegmentationPredictor(settings.model_path, generated_dir),
    )


def create_app(inference_service: InferenceService | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name)
    artifacts_dir = settings.project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=artifacts_dir), name="artifacts")
    runtime_inference_service = inference_service
    if runtime_inference_service is None:
        runtime_inference_service = build_default_inference_service(settings, artifacts_dir)
    app.include_router(
        build_router(runtime_inference_service)
    )
    return app


app = create_app()
