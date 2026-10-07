import random
import logging
from typing import Dict, Any, List, Optional
from app.models.registry import ModelRegistry
from app.routing.scoring import calculate_score
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelRouter:
    def __init__(self, registry: ModelRegistry, manager=None):
        self.registry = registry
        self.manager = manager
        self.weights = {
            "latency_weight": settings.app_config.routing.latency_weight if settings.app_config else 0.4,
            "accuracy_weight": settings.app_config.routing.accuracy_weight if settings.app_config else 0.6
        }
        self.canary_percentage = settings.app_config.canary.percentage if settings.app_config else 5

    def route_request(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        model_name = request_params.get("model")
        if not model_name:
            raise ValueError("Model name is required")
            
        candidates = self.registry.get_active_models(model_name)
        if not candidates:
            raise RuntimeError(f"No active candidates for model: {model_name}")
            
        # 1. Check Circuit Breaker health via manager
        healthy_candidates = []
        for c in candidates:
            c_id = f"{c['model_name']}:{c['version']}"
            if self.manager:
                backend = self.manager.get_model(c_id)
                # If health isn't HEALTHY (e.g. breaker is OPEN), skip it
                if backend and backend.health() != "HEALTHY":
                    logger.debug(f"Skipping {c_id} due to unhealthy backend state")
                    continue
            healthy_candidates.append(c)
            
        if not healthy_candidates:
            raise RuntimeError(f"No healthy models available for {model_name}")

        # 2. Canary Tier Selection
        stable_models = [m for m in healthy_candidates if m.get('metadata', {}).get('tier') == 'stable']
        canary_models = [m for m in healthy_candidates if m.get('metadata', {}).get('tier') == 'canary']
        
        selected_tier_models = healthy_candidates
        if stable_models and canary_models:
            roll = random.uniform(0, 100)
            if roll <= self.canary_percentage:
                selected_tier_models = canary_models
            else:
                selected_tier_models = stable_models
        elif stable_models:
            selected_tier_models = stable_models
        elif canary_models:
            selected_tier_models = canary_models
            
        # 3. Score variants within the chosen tier
        best_candidate = None
        highest_score = -1.0
        
        for candidate in selected_tier_models:
            score = calculate_score(candidate, request_params, self.weights)
            # P2-8: Hard filter over-budget models is applied inside calculate_score by returning -1 for invalid
            if score > highest_score:
                highest_score = score
                best_candidate = candidate
                
        # If all exceeded budget, it would return -1, so try best effort fallback (just pick best scored, or any)
        if highest_score < 0:
            logger.warning(f"All candidates exceeded latency budget for {model_name}. Proceeding with best-effort.")
            # Just take the first one
            best_candidate = selected_tier_models[0]
            best_candidate["budget_violated"] = True
            
        if not best_candidate:
            raise RuntimeError("No suitable model found for the given constraints.")
            
        return best_candidate
