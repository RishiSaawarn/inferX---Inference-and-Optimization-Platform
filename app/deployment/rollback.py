import logging
import asyncio
import time
from typing import Dict, Any

from app.core.config import settings
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

class RollbackEvaluator:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.latency_multiplier = settings.app_config.rollback.latency_multiplier if settings.app_config else 1.25
        self.max_error_rate = settings.app_config.rollback.max_error_rate if settings.app_config else 0.02
        self.min_samples = settings.app_config.rollback.min_samples if settings.app_config else 100
        self._running = False
        
        # In a real system, metrics come from Prometheus or similar TSDB.
        # Here we mock the data source.
        self.mock_metrics_store: Dict[str, Dict[str, Any]] = {}

    def evaluate(self, model_name: str, stable_version: str, canary_version: str) -> None:
        stable_id = f"{model_name}:{stable_version}"
        canary_id = f"{model_name}:{canary_version}"
        
        stable_metrics = self.mock_metrics_store.get(stable_id, {"p95_latency_ms": 50.0, "error_rate": 0.0, "samples": 500})
        canary_metrics = self.mock_metrics_store.get(canary_id, {"p95_latency_ms": 0.0, "error_rate": 0.0, "samples": 0})
        
        if canary_metrics["samples"] < self.min_samples:
            return # Wait for more data

        stable_p95 = stable_metrics.get("p95_latency_ms", 0.0)
        canary_p95 = canary_metrics.get("p95_latency_ms", 0.0)
        canary_error = canary_metrics.get("error_rate", 0.0)

        reason = None
        if stable_p95 > 0 and canary_p95 > stable_p95 * self.latency_multiplier:
            reason = f"P95 latency regression (Stable: {stable_p95}, Canary: {canary_p95})"
        elif canary_error > self.max_error_rate:
            reason = f"Error rate exceeded threshold (Canary: {canary_error})"

        if reason:
            logger.warning(f"ROLLBACK INITIATED for {canary_id}. Reason: {reason}")
            self.registry.update_status(model_name, canary_version, "FAILED")
        elif canary_metrics["samples"] > self.min_samples * 5: # Arbitrary promotion logic
            logger.info(f"PROMOTING {canary_id} to stable!")
            metadata = self.registry.get_active_models(model_name)
            for row in metadata:
                if row['version'] == canary_version:
                    row['metadata']['tier'] = 'stable'
                    self.registry.register_model(model_name, canary_version, row['runtime'], 
                                                 row['device'], row['precision'], row['metadata'], row['artifact_path'])
            # And demote the old stable
            self.registry.update_status(model_name, stable_version, "INACTIVE")

    async def _evaluation_loop(self):
        while self._running:
            try:
                # Find models with both stable and canary
                all_active = self.registry.get_all_active_models()
                by_model = {}
                for row in all_active:
                    by_model.setdefault(row['model_name'], []).append(row)
                    
                for model_name, rows in by_model.items():
                    stable = next((r for r in rows if r.get('metadata', {}).get('tier') == 'stable'), None)
                    canary = next((r for r in rows if r.get('metadata', {}).get('tier') == 'canary'), None)
                    
                    if stable and canary:
                        self.evaluate(model_name, stable['version'], canary['version'])
                        
            except Exception as e:
                logger.error(f"Error in rollback evaluation loop: {e}")
                
            await asyncio.sleep(10) # Check every 10 seconds

    def start(self):
        if not self._running:
            self._running = True
            asyncio.create_task(self._evaluation_loop())
            
    def stop(self):
        self._running = False
