import logging
import random
from typing import Any

from app.core.config import settings
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class CanaryRouter:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.canary_percentage = (
            settings.app_config.canary.percentage if settings.app_config else 5
        )

    def route(self, model_name: str, request_params: dict[str, Any]) -> dict[str, Any]:
        active_models = self.registry.get_active_models(model_name)

        stable_models = [
            m for m in active_models if m.get("metadata", {}).get("tier") == "stable"
        ]
        canary_models = [
            m for m in active_models if m.get("metadata", {}).get("tier") == "canary"
        ]

        # Fallback to any active if tier isn't specified
        if not stable_models and not canary_models:
            if not active_models:
                raise RuntimeError(f"No models available for {model_name}")
            return random.choice(active_models)

        if stable_models and canary_models:
            roll = random.uniform(0, 100)
            if roll <= self.canary_percentage:
                logger.info(f"Routing request to CANARY for {model_name}")
                return random.choice(canary_models)
            else:
                return random.choice(stable_models)
        elif stable_models:
            return random.choice(stable_models)
        else:
            return random.choice(canary_models)
