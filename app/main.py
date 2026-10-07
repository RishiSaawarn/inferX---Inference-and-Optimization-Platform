import logging

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="InferX",
    description="Production AI Inference & Optimization Platform",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    logger.info(
        "Starting InferX Application",
        extra={"event": "startup", "env": settings.app_env},
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "env": settings.app_env}
