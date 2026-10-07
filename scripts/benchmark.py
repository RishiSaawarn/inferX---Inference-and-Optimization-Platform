import time
import numpy as np
import onnxruntime as ort
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_model(model_path: str, batch_sizes=[1, 4, 16], report_output="benchmark_report.json"):
    logger.info(f"Benchmarking {model_path}...")
    
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    results = {}
    
    for bs in batch_sizes:
        # P2-16: Fix tensor shape dimension mapping
        dummy_input = np.random.randn(bs, 3, 224, 224).astype(np.float32)
        
        # Warmup (P2-16: proper count)
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
            
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            latencies.append((time.perf_counter() - start) * 1000)
            
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        logger.info(f"Batch Size {bs}: P50 {p50:.2f}ms, P95 {p95:.2f}ms")
        results[f"bs_{bs}"] = {"p50_ms": p50, "p95_ms": p95}
        
    with open(report_output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {report_output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--report", type=str, default="benchmark_report.json")
    args = parser.parse_args()
    benchmark_model(args.model, report_output=args.report)
