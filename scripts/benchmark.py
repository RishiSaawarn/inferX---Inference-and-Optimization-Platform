import argparse
import logging
import time

import numpy as np

from app.inference.onnx_backend import ONNXBackend
from app.inference.pytorch_backend import PyTorchBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")


def benchmark_model(backend, iterations=100, batch_size=1):
    logger.info(f"Loading {backend.__class__.__name__}...")
    backend.load()

    logger.info("Warming up...")
    backend.warmup()

    # Assuming ImageNet shape
    if (
        hasattr(backend, "device")
        and hasattr(backend.device, "type")
        and backend.device.type == "cpu"
    ):
        import torch

        dummy_input = torch.randn(batch_size, 3, 224, 224)
    else:
        dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    latencies = []
    logger.info(f"Running {iterations} iterations with batch_size={batch_size}...")
    for _ in range(iterations):
        start = time.perf_counter()
        backend.predict(dummy_input)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = (batch_size * iterations) / (np.sum(latencies) / 1000)

    logger.info("=== Benchmark Results ===")
    logger.info(f"P50 Latency: {p50:.2f} ms")
    logger.info(f"P95 Latency: {p95:.2f} ms")
    logger.info(f"P99 Latency: {p99:.2f} ms")
    logger.info(f"Throughput:  {throughput:.2f} req/s")

    backend.unload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, required=True, choices=["pytorch", "onnx"]
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.backend == "pytorch":
        backend = PyTorchBackend(args.model_name_or_path, device=args.device)
    else:
        backend = ONNXBackend(args.model_name_or_path, device=args.device)

    benchmark_model(backend, args.iterations, args.batch_size)
