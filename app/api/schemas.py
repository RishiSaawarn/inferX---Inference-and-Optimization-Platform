from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict, Any

class InferenceRequest(BaseModel):
    model: str = Field(..., description="Name of the model to use")
    latency_budget_ms: float = Field(default=100.0, gt=0, description="Maximum acceptable latency in ms")
    accuracy_priority: Literal["low", "balanced", "high"] = Field(default="balanced", description="Accuracy vs Latency priority tradeoff")
    input_data: Union[List[Any], str] = Field(..., description="Input data for inference (base64 Image string or nested list)")

class InferenceTimings(BaseModel):
    queue_ms: float
    exec_ms: float
    total_ms: float

class InferenceResponse(BaseModel):
    model: str
    version: str
    backend: str
    precision: str
    predictions: List[Dict[str, Any]]
    timings: InferenceTimings
    cache_hit: bool
    routed_to_canary: bool
