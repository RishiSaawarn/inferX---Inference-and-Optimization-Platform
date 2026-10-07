import argparse
import logging
import json
import os
from pathlib import Path
from app.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_models")

def build_models():
    # In a full run, we would call export_onnx.py and quantize.py here.
    # For now, we will just register the standard downloaded models as PyTorch FP32.
    # The actual export processes are handled by the specific scripts.
    
    registry = ModelRegistry()
    
    models = ["resnet18", "mobilenet_v3_small"]
    
    for m in models:
        # Register PyTorch FP32 variant
        metadata = {
            "tier": "stable",
            "p95_latency_ms": 50.0,
            "error_rate": 0.0,
            "accuracy": 0.9,
            "capabilities": ["image_classification"]
        }
        
        registry.register_model(
            model_name=m, 
            version="v1.0",
            runtime="pytorch",
            device="cpu", # Could be cuda if available
            precision="fp32",
            metadata=metadata,
            artifact_path=""  # PyTorch backend loads from torchvision weights internally for this demo
        )
        
        # Activate it
        registry.update_status(m, "v1.0", "ACTIVE")
        logger.info(f"Activated {m}:v1.0 (PyTorch FP32)")
        
        # We can also seed a canary version
        canary_meta = metadata.copy()
        canary_meta["tier"] = "canary"
        registry.register_model(
            model_name=m, 
            version="v2.0-canary",
            runtime="pytorch",
            device="cpu",
            precision="fp32",
            metadata=canary_meta,
            artifact_path=""
        )
        registry.update_status(m, "v2.0-canary", "ACTIVE")
        logger.info(f"Activated {m}:v2.0-canary (PyTorch FP32)")

if __name__ == "__main__":
    build_models()
