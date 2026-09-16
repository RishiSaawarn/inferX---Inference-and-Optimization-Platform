# Trade-Offs

## ONNX vs PyTorch
PyTorch is easier to debug and develop, but ONNX Runtime typically offers better CPU inference performance and portability.

## Batching vs Latency
Dynamic batching increases throughput but adds a fixed wait time (e.g., 5ms) to early requests, increasing their latency. The trade-off is managed by adjusting `max_wait_ms`.

## Caching vs Freshness
Redis response caching reduces tail latency for repeated inputs but requires careful invalidation strategies.

## Synchronous vs Asynchronous Inference
Model inference is computationally intensive and synchronous. We run inference in a `ThreadPoolExecutor` to avoid blocking the `asyncio` event loop.
