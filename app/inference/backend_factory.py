from typing import Dict, Any
from app.inference.base import InferenceBackend
from app.inference.pytorch_backend import PyTorchBackend
from app.inference.onnx_backend import ONNXBackend

class BackendFactory:
    @staticmethod
    def create(registry_row: Dict[str, Any]) -> InferenceBackend:
        runtime = registry_row.get("runtime", "").lower()
        model_name = registry_row.get("model_name")
        device = registry_row.get("device", "cpu")
        artifact_path = registry_row.get("artifact_path", "")
        
        if not model_name:
            raise ValueError("Row requires model_name")
            
        if runtime == "pytorch":
            return PyTorchBackend(model_name=model_name, device=device)
        elif runtime == "onnxruntime":
            if not artifact_path:
                raise ValueError(f"artifact_path required for ONNX models, got: {registry_row}")
            return ONNXBackend(model_path=artifact_path, device=device)
        else:
            raise ValueError(f"Unsupported runtime: {runtime}")
