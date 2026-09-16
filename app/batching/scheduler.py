from typing import Dict, Any
from app.batching.queue import BatchQueue
from app.inference.manager import ModelManager
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class BatchScheduler:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.queues: Dict[str, BatchQueue] = {}
        self.enabled = settings.app_config.batching.enabled if settings.app_config else False
        self.max_batch_size = settings.app_config.batching.max_batch_size if settings.app_config else 16
        self.max_wait_ms = settings.app_config.batching.max_wait_ms if settings.app_config else 5
        
    def get_or_create_queue(self, model_id: str) -> BatchQueue:
        if model_id not in self.queues:
            backend = self.manager.get_model(model_id)
            if not backend:
                raise KeyError(f"Model {model_id} not loaded in manager")
            queue = BatchQueue(backend, self.max_batch_size, self.max_wait_ms)
            queue.start()
            self.queues[model_id] = queue
            logger.info(f"Created batch queue for model {model_id}")
        return self.queues[model_id]
        
    async def predict_async(self, model_id: str, input_data: Any) -> Any:
        if not self.enabled:
            import asyncio
            backend = self.manager.get_model(model_id)
            if not backend:
                raise KeyError(f"Model {model_id} not found")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, backend.predict, input_data)
            
        queue = self.get_or_create_queue(model_id)
        return await queue.enqueue(input_data)
        
    async def shutdown(self):
        for model_id, queue in self.queues.items():
            await queue.stop()
