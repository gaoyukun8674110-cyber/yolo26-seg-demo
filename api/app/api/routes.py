from fastapi import APIRouter, File, UploadFile

from api.app.core.settings import get_settings
from api.app.schemas.predict import PredictResponse
from api.app.services.content_store import ContentStore
from api.app.services.inference import InferenceService


def build_router(inference_service: InferenceService) -> APIRouter:
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

    @router.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)) -> PredictResponse:
        result = inference_service.predict(await file.read(), file.filename or "upload.png")
        return PredictResponse(
            has_defect=result.has_defect,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            overlay_url=f"/artifacts/generated/{result.overlay_filename}",
        )

    return router
