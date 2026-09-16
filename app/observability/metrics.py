from prometheus_client import Counter, Histogram, Gauge

inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["model", "version", "backend", "device", "precision", "status"]
)

inference_errors_total = Counter(
    "inference_errors_total",
    "Total number of inference errors",
    ["model", "version", "backend", "device", "precision"]
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Total inference latency in seconds",
    ["model", "version", "backend"]
)

inference_queue_latency_seconds = Histogram(
    "inference_queue_latency_seconds",
    "Time spent in batching queue",
    ["model"]
)

inference_execution_latency_seconds = Histogram(
    "inference_execution_latency_seconds",
    "Actual model execution time",
    ["model", "backend"]
)

model_load_time_seconds = Histogram(
    "model_load_time_seconds",
    "Time to load model into memory",
    ["model", "backend"]
)

model_health = Gauge(
    "model_health",
    "Model health status (1=healthy, 0=unhealthy)",
    ["model", "version"]
)

cache_hits_total = Counter("cache_hits_total", "Total cache hits", ["model"])
cache_misses_total = Counter("cache_misses_total", "Total cache misses", ["model"])
batch_size = Histogram("batch_size", "Batch size for inference", ["model"])
queue_depth = Gauge("queue_depth", "Current number of requests in queue", ["model"])
active_requests = Gauge("active_requests", "Current active requests", ["model"])
rollback_total = Counter("rollback_total", "Total automatic rollbacks", ["model"])
canary_requests_total = Counter("canary_requests_total", "Total requests routed to canary", ["model"])
slo_violations_total = Counter("slo_violations_total", "Total SLO violations", ["model"])
