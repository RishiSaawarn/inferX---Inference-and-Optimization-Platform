from abc import ABC, abstractmethod
from typing import Any, Dict

class InferenceBackend(ABC):
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Perform warmup inference."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on the given input."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory and free resources."""
        pass

    @abstractmethod
    def health(self) -> str:
        """Return the health status of the backend."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the backend."""
        pass
