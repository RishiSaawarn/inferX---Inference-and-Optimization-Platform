from app.routing.scoring import calculate_score

def test_scoring_budget():
    candidate = {"metadata": {"p95_latency_ms": 150}}
    request_params = {"latency_budget_ms": 100}
    score = calculate_score(candidate, request_params, {})
    assert score == -1.0
    
def test_scoring_weights():
    candidate = {"metadata": {"p95_latency_ms": 50, "accuracy": 0.9}}
    request_params = {"latency_budget_ms": 100, "accuracy_priority": "high"}
    score = calculate_score(candidate, request_params, {})
    # 0.8 * 0.9 + 0.2 * 0.5 = 0.72 + 0.1 = 0.82
    assert score > 0.8
