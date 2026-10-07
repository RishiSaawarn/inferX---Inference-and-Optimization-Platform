import multiprocessing
from typing import Any

import torch
from torch import nn
from torchvision import models

from app.inference.base import InferenceBackend


class PyTorchBackend(InferenceBackend):
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model: nn.Module | None = None

        # Limit CPU threads to avoid oversubscription (P2-15)
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, multiprocessing.cpu_count() // 2))

    def load(self) -> None:
        if self.model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
        elif self.model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model.to(self.device)
        model.eval()
        self.model = model

    def warmup(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.inference_mode():
            for _ in range(3):
                self.model(dummy_input)

    def predict(self, input_data: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # Ensure input is a tensor and on the correct device
        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        input_data = input_data.to(self.device)
        with torch.inference_mode():
            output = self.model(input_data)
        return output

    def unload(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def health(self) -> str:
        return "HEALTHY" if self.model is not None else "STARTING"

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "runtime": "pytorch",
            "device": self.device.type,
            "precision": "fp32",
        }
