from typing import Any


def calculate_score(
    model_metadata: dict[str, Any],
    request_params: dict[str, Any],
    weights: dict[str, float],
) -> float:
    # Extract embedded metadata which contains performance metrics
    meta = model_metadata.get("metadata", {})

    latency_budget = request_params.get("latency_budget_ms", 100)
    p95_latency = meta.get("p95_latency_ms", 50)

    latency_score = 0.0
    if p95_latency <= latency_budget:
        # Score higher for lower latency relative to budget
        latency_score = 1.0 - (p95_latency / latency_budget)

    accuracy = meta.get("accuracy", 0.5)
    accuracy_score = accuracy

    score = latency_score * weights.get(
        "latency_weight", 0.4
    ) + accuracy_score * weights.get("accuracy_weight", 0.6)
    return score
