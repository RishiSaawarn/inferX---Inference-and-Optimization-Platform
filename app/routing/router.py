from typing import Dict, Any, List
from app.models.registry import ModelRegistry
from app.routing.scoring import calculate_score
from app.core.config import settings

class ModelRouter:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.weights = {
            "latency_weight": settings.app_config.routing.latency_weight if settings.app_config else 0.4,
            "accuracy_weight": settings.app_config.routing.accuracy_weight if settings.app_config else 0.6
        }

    def route_request(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        model_name = request_params.get("model")
        if not model_name:
            raise ValueError("Model name is required")
            
        candidates = self.registry.get_active_models(model_name)
        if not candidates:
            raise RuntimeError(f"No active candidates for model: {model_name}")
            
        best_candidate = None
        highest_score = -1.0
        
        for candidate in candidates:
            score = calculate_score(candidate, request_params, self.weights)
            if score > highest_score:
                highest_score = score
                best_candidate = candidate
                
        if not best_candidate:
            raise RuntimeError("No suitable model found for the given constraints.")
            
        return dict(best_candidate)
