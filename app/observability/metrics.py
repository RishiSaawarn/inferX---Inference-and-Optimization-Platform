from prometheus_client import Counter, Histogram, Gauge

# P1-6: Implement actual metrics recording with fine-grained buckets for ML

# Buckets from 1ms up to 5 seconds
LATENCY_BUCKETS = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")
)

# Request counters
REQUEST_COUNT = Counter(
    "inferx_requests_total",
    "Total incoming inference requests",
    ["model_name", "status"]
)

# Inference Latency
INFERENCE_LATENCY = Histogram(
    "inferx_inference_duration_seconds",
    "Model inference execution time",
    ["model_id", "backend", "precision"],
    buckets=LATENCY_BUCKETS
)

# Overall API Latency
API_LATENCY = Histogram(
    "inferx_api_duration_seconds",
    "Total API request duration",
    ["model_name", "status"],
    buckets=LATENCY_BUCKETS
)

# Cache hits/misses
CACHE_HITS = Counter(
    "inferx_cache_hits_total",
    "Total response cache hits versus misses",
    ["model_name", "hit"]
)

# Fallback/Retry counters
RESILIENCE_EVENTS = Counter(
    "inferx_resilience_events_total",
    "Times retries or fallbacks were triggered",
    ["model_id", "type"]
)

# Queue depths
QUEUE_SIZE = Gauge(
    "inferx_queue_size",
    "Current number of requests waiting in the batch queue",
    ["model_id"]
)

def record_request(model_name: str, status: str):
    REQUEST_COUNT.labels(model_name=model_name, status=status).inc()

def record_inference_time(model_id: str, backend: str, precision: str, duration_sec: float):
    INFERENCE_LATENCY.labels(model_id=model_id, backend=backend, precision=precision).observe(duration_sec)

def record_api_time(model_name: str, status: str, duration_sec: float):
    API_LATENCY.labels(model_name=model_name, status=status).observe(duration_sec)

def record_cache(model_name: str, hit: bool):
    CACHE_HITS.labels(model_name=model_name, hit=str(hit)).inc()
