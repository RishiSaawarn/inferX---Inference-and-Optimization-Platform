# InferX Architecture

InferX is designed as a resilient, high-performance API Gateway and model serving layer.

```mermaid
graph TD
    Client[Client] --> Gateway[API Gateway (FastAPI)]
    Gateway --> Validator[Request Validator]
    Validator --> Router[Model Router]
    
    Router --> |Hardware, Latency Budget, Health| Backend_Selector{Select Backend}
    Backend_Selector --> PyTorch[PyTorch CPU/GPU]
    Backend_Selector --> ONNX[ONNX CPU/GPU]
    
    PyTorch --> Validator_Out[Result Validator]
    ONNX --> Validator_Out
    Validator_Out --> Cache[Redis Cache]
    Cache --> Response[Response]
```

## Component Responsibilities
- **Model Router**: Evaluates candidate models against latency budgets, health scores, and accuracy requirements.
- **Batch Scheduler**: Assembles concurrent requests into optimal batches.
- **Model Registry**: PostgreSQL-backed persistent storage for model metadata and versions.
- **Circuit Breaker**: Prevents cascading failures by opening circuits on repeatedly failing backends.
