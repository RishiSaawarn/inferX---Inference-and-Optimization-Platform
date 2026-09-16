import pytest
import torch
import numpy as np
from app.inference.pytorch_backend import PyTorchBackend
from app.inference.manager import ModelManager

def test_pytorch_backend_load():
    backend = PyTorchBackend("resnet18", device="cpu")
    assert backend.health() == "STARTING"
    backend.load()
    assert backend.health() == "HEALTHY"
    
    # Test warmup
    backend.warmup()
    
    # Test predict
    dummy_input = torch.randn(1, 3, 224, 224)
    output = backend.predict(dummy_input)
    assert output is not None
    assert output.shape == (1, 1000)
    
    backend.unload()
    assert backend.health() == "STARTING"

def test_model_manager():
    manager = ModelManager()
    backend = PyTorchBackend("resnet18", device="cpu")
    
    manager.register_model("resnet18", backend)
    assert manager.get_model("resnet18") is not None
    
    manager.load_model("resnet18")
    assert manager.get_model("resnet18").health() == "HEALTHY"
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output = manager.predict("resnet18", dummy_input)
    assert output is not None
    
    manager.unload_model("resnet18")
