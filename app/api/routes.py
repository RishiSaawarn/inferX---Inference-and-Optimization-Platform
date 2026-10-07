import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from app.api.schemas import InferenceRequest, InferenceResponse, InferenceTimings
from app.observability.metrics import record_request, record_api_time, RESILIENCE_EVENTS
from app.cache.cache import ResponseCache
from app.resilience.retry import with_retry
from app.resilience.fallback import execute_with_fallback
from app.core.exceptions import BudgetExceeded
from app.inference.preprocessing import preprocess_request
from app.inference.postprocessing import postprocess_output
from app.core.profiler import RequestProfiler

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/inference", response_model=InferenceResponse)
async def predict(request: Request, payload: InferenceRequest):
    # This is currently a stub that we will extend with full caching, 
    # routing, and circuit breakers in later P1 steps.
    start_time = time.monotonic()
    
    # 1. Routing
    app_state = request.app.state
    if not hasattr(app_state, "model_router"):
        raise HTTPException(status_code=503, detail="Router not initialized")
        
    try:
        route_params = {
            "model": payload.model,
            "latency_budget_ms": payload.latency_budget_ms,
            "accuracy_priority": payload.accuracy_priority
        }
        # In P1-5 this will be composed, for now use existing ModelRouter
        best_candidate = app_state.model_router.route_request(route_params)
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
        
    if not best_candidate:
        raise HTTPException(status_code=404, detail="No suitable model version found")

    model_id = f"{best_candidate['model_name']}:{best_candidate['version']}"
    
    # Check cache first
    response_cache = getattr(app_state, "response_cache", None)
    if response_cache:
        cached_result = await response_cache.get(model_id, payload.input_data)
        if cached_result:
            cached_result["timings"]["total_ms"] = (time.monotonic() - start_time) * 1000.0
            cached_result["cache_hit"] = True
            record_api_time(payload.model, "success", time.monotonic() - start_time)
            return InferenceResponse(**cached_result)
        
    if not hasattr(app_state, "batch_scheduler"):
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
        
    profiler = RequestProfiler()
        
    # Execute with resilience patterns
    try:
        # Define the execution block to be retried/fallback
        async def _do_inference():
            with profiler.measure("preprocessing"):
                # Run CPU-bound preprocessing in threadpool if it blocks, but here we just call it directly for now or assume it's fast
                tensor_input = preprocess_request(payload.input_data)
                
            exec_start = time.monotonic()
            raw_result = await app_state.batch_scheduler.predict_async(model_id, tensor_input)
            exec_ms = (time.monotonic() - exec_start) * 1000.0
            
            with profiler.measure("postprocessing"):
                final_result = postprocess_output(raw_result)
                
            return final_result, exec_ms
            
        async def _fallback(exc: Exception):
            RESILIENCE_EVENTS.labels(model_id=model_id, type="fallback").inc()
            logger.warning(f"Fallback triggered for {model_id} due to {exc}")
            # Mock fallback result
            return [{"class_id": 0, "score": 0.0}], 0.0
            
        # Protect with retry and fallback
        result_tuple = await execute_with_fallback(
            with_retry(_do_inference, max_retries=1, base_delay=0.1),
            fallback_func=_fallback
        )
        
        predictions, exec_ms = result_tuple
        
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        record_api_time(payload.model, "error", time.monotonic() - start_time)
        raise HTTPException(status_code=500, detail=str(e))

    total_time_ms = (time.monotonic() - start_time) * 1000.0

    resp = InferenceResponse(
        model=best_candidate["model_name"],
        version=best_candidate["version"],
        backend=best_candidate.get("runtime", "unknown"),
        precision=best_candidate.get("precision", "unknown"),
        predictions=predictions,
        timings=InferenceTimings(
            queue_ms=0.0,
            exec_ms=exec_ms,
            total_ms=total_time_ms
        ),
        cache_hit=False,
        routed_to_canary=best_candidate.get("metadata", {}).get("tier") == "canary"
    )
    
    # Save to cache asynchronously
    if response_cache:
        import asyncio
        asyncio.create_task(response_cache.set(model_id, payload.input_data, resp.dict()))
    
    record_api_time(payload.model, "success", total_time_ms / 1000.0)
    record_request(payload.model, "success")
    
    return resp
