import logging

from app.core.config import settings
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RollbackEvaluator:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.latency_multiplier = (
            settings.app_config.rollback.latency_multiplier
            if settings.app_config
            else 1.25
        )
        self.max_error_rate = (
            settings.app_config.rollback.max_error_rate if settings.app_config else 0.02
        )

    def evaluate(
        self,
        stable_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        model_name: str,
        candidate_version: str,
    ) -> bool:
        """
        Evaluate if a candidate model should be rolled back.
        Returns True if rolled back, False otherwise.
        """
        stable_p95 = stable_metrics.get("p95_latency_ms", 0.0)
        candidate_p95 = candidate_metrics.get("p95_latency_ms", 0.0)
        candidate_error = candidate_metrics.get("error_rate", 0.0)

        reason = None
        if stable_p95 > 0 and candidate_p95 > stable_p95 * self.latency_multiplier:
            reason = f"P95 latency regression (Stable: {stable_p95}, Candidate: {candidate_p95})"
        elif candidate_error > self.max_error_rate:
            reason = f"Error rate exceeded threshold (Candidate: {candidate_error})"

        if reason:
            logger.warning(
                f"ROLLBACK INITIATED for {model_name}:{candidate_version}. Reason: {reason}"
            )
            # Perform rollback by marking candidate as FAILED
            self.registry.update_status(model_name, candidate_version, "FAILED")
            return True

        return False
