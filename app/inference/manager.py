from typing import Dict, Optional, Any
from app.inference.base import InferenceBackend

class ModelManager:
    def __init__(self):
        self.models: Dict[str, InferenceBackend] = {}
        
    def register_model(self, model_id: str, backend: InferenceBackend) -> None:
        self.models[model_id] = backend
        
    def load_model(self, model_id: str) -> None:
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not registered")
        backend = self.models[model_id]
        backend.load()
        backend.warmup()
        
    def unload_model(self, model_id: str) -> None:
        if model_id in self.models:
            self.models[model_id].unload()
            
    def get_model(self, model_id: str) -> Optional[InferenceBackend]:
        return self.models.get(model_id)

    def predict(self, model_id: str, input_data: Any) -> Any:
        backend = self.get_model(model_id)
        if not backend:
            raise KeyError(f"Model {model_id} not found")
        if backend.health() != "HEALTHY":
            raise RuntimeError(f"Model {model_id} is not healthy")
        return backend.predict(input_data)
