from fastapi import FastAPI

from api.app.api.routes import router
from api.app.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.service_name)
    app.include_router(router)
    return app


app = create_app()
