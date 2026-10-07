from abc import ABC, abstractmethod
from typing import Any


class InferenceBackend(ABC):

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def warmup(self) -> None:
        """Perform warmup inference."""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on the given input."""

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory and free resources."""

    @abstractmethod
    def health(self) -> str:
        """Return the health status of the backend."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return metadata about the backend."""
