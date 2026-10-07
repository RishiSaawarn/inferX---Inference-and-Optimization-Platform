import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.observability.tracing import setup_tracing
from app.api.routes import router as api_router
from app.models.registry import ModelRegistry
from app.routing.router import ModelRouter
from app.inference.manager import ModelManager
from app.inference.backend_factory import BackendFactory
from app.batching.scheduler import BatchScheduler
from app.deployment.rollback import RollbackEvaluator

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting InferX Application", extra={"event": "startup", "env": settings.app_env})
    setup_tracing(app)
    
    registry = ModelRegistry()
    model_router = ModelRouter(registry)
    manager = ModelManager()
    
    active_rows = registry.get_all_active_models()
    for row in active_rows:
        try:
            model_id = f"{row['model_name']}:{row['version']}"
            backend = BackendFactory.create(row)
            manager.register_model(model_id, backend)
            manager.load_model(model_id)
            logger.info(f"Loaded active model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load model {row.get('model_name')}: {e}")
            
    batch_scheduler = BatchScheduler(manager)
    
    rollback_evaluator = RollbackEvaluator(registry)
    rollback_evaluator.start()
    
    app.state.registry = registry
    app.state.model_router = model_router
    app.state.manager = manager
    app.state.batch_scheduler = batch_scheduler
    app.state.rollback_evaluator = rollback_evaluator
    
    yield
    
    logger.info("Shutting down InferX Application", extra={"event": "shutdown"})
    rollback_evaluator.stop()
    await batch_scheduler.shutdown()
    for model_id in list(manager.models.keys()):
        manager.unload_model(model_id)

app = FastAPI(
    title="InferX",
    description="Production AI Inference & Optimization Platform",
    version="0.1.0",
    lifespan=lifespan
)

from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(api_router, prefix="/v1")

@app.get("/health/live")
def health_live():
    return {"status": "ok", "env": settings.app_env}
    
@app.get("/health/ready")
def health_ready(request: FastAPI):
    # TODO: P3-5 Readiness
    return {"status": "ok"}
