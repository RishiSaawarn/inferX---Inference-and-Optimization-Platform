import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_score(candidate: Dict[str, Any], request_params: Dict[str, Any], global_weights: Dict[str, float]) -> float:
    metadata = candidate.get("metadata", {})
    
    # Request constraints
    budget_ms = request_params.get("latency_budget_ms", 100.0)
    acc_priority = request_params.get("accuracy_priority", "balanced")
    
    p95_latency = metadata.get("p95_latency_ms", budget_ms)
    accuracy = metadata.get("accuracy", 0.0)
    
    # P2-8: Hard-filter over-budget models (return -1 so router ignores them)
    if p95_latency > budget_ms:
        return -1.0
        
    # P2-9: Map accuracy priority to weights
    if acc_priority == "high":
        w_acc = 0.8
        w_lat = 0.2
    elif acc_priority == "low":
        w_acc = 0.2
        w_lat = 0.8
    else:
        # Balanced usually uses the global fallback, but we use defaults 0.5/0.5
        w_acc = global_weights.get("accuracy_weight", 0.5)
        w_lat = global_weights.get("latency_weight", 0.5)
        
    # Normalize weights just in case
    total_w = w_acc + w_lat
    w_acc /= total_w
    w_lat /= total_w
    
    # Score calculation (higher is better). Latency is better when smaller, so we invert it.
    # Latency score approaches 1 as p95 approaches 0, and approaches 0 as p95 approaches budget
    latency_score = max(0.0, 1.0 - (p95_latency / budget_ms))
    accuracy_score = accuracy # assuming it's usually between 0-1
    
    final_score = (w_acc * accuracy_score) + (w_lat * latency_score)
    # Give a tiny boost to models with zero error rate (P2-9 extra robust)
    error_rate = metadata.get("error_rate", 0.0)
    if error_rate == 0.0:
        final_score += 0.05
        
    return final_score
