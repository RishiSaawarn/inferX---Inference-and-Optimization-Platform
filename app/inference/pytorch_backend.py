from typing import Any, Dict
import torch
import torchvision.models as models
from app.inference.base import InferenceBackend

class PyTorchBackend(InferenceBackend):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None

    def load(self) -> None:
        if self.model_name == "mobilenet_v3_small":
            # Using weights instead of pretrained=True to avoid warnings if possible, 
            # but pretrained=True is simpler for this.
            self.model = models.mobilenet_v3_small(pretrained=True)
        elif self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        self.model.to(self.device)
        self.model.eval()

    def warmup(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy_input)

    def predict(self, input_data: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # Ensure input is a tensor and on the correct device
        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        input_data = input_data.to(self.device)
        with torch.no_grad():
            output = self.model(input_data)
        return output

    def unload(self) -> None:
        self.model = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def health(self) -> str:
        return "HEALTHY" if self.model is not None else "STARTING"

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "runtime": "pytorch",
            "device": self.device.type,
            "precision": "fp32"
        }
