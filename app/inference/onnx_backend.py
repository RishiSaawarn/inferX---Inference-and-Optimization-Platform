from typing import Any

import numpy as np
import onnxruntime as ort

from app.inference.base import InferenceBackend


class ONNXBackend(InferenceBackend):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.session = None

    def load(self) -> None:
        providers = ["CPUExecutionProvider"]
        if (
            self.device == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers = ["CUDAExecutionProvider"] + providers

        self.session = ort.InferenceSession(self.model_path, providers=providers)

    def warmup(self) -> None:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        # Determine input shape from session
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic batch size
        shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]
        dummy_input = np.random.randn(*shape).astype(np.float32)

        for _ in range(3):
            self.session.run(None, {input_name: dummy_input})

    def predict(self, input_data: Any) -> Any:
        if self.session is None:
            raise RuntimeError("Model not loaded")
        input_name = self.session.get_inputs()[0].name
        if not isinstance(input_data, np.ndarray):
            # Try converting assuming it's a torch tensor or compatible format
            try:
                input_data = input_data.numpy()
            except AttributeError:
                input_data = np.array(input_data, dtype=np.float32)

        output = self.session.run(None, {input_name: input_data})
        return output[0]

    def unload(self) -> None:
        self.session = None

    def health(self) -> str:
        return "HEALTHY" if self.session is not None else "STARTING"

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "runtime": "onnxruntime",
            "device": self.device,
            "providers": self.session.get_providers() if self.session else [],
        }
