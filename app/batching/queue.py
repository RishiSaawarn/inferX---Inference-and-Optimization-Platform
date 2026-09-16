import asyncio
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BatchQueue:
    def __init__(self, backend, max_batch_size: int, max_wait_ms: int):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task = None

    def start(self):
        if not self.task:
            self.task = asyncio.create_task(self._process_queue())

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _process_queue(self):
        while True:
            batch = []
            futures = []
            try:
                # Wait for at least one item
                req, future = await self.queue.get()
                batch.append(req)
                futures.append(future)
                
                # Try to gather more items up to max_batch_size
                deadline = time.time() + self.max_wait_ms
                while len(batch) < self.max_batch_size:
                    timeout = deadline - time.time()
                    if timeout <= 0:
                        break
                    try:
                        req, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(req)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                        
                if batch:
                    # Execute batch
                    try:
                        import torch
                        import numpy as np
                        
                        if isinstance(batch[0], torch.Tensor):
                            batched_input = torch.cat(batch, dim=0)
                        elif isinstance(batch[0], np.ndarray):
                            batched_input = np.concatenate(batch, axis=0)
                        else:
                            batched_input = batch
                            
                        loop = asyncio.get_event_loop()
                        results = await loop.run_in_executor(None, self.backend.predict, batched_input)
                        
                        # Dispatch results
                        for i, future in enumerate(futures):
                            if not future.done():
                                future.set_result(results[i:i+1])
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue error: {e}")

    async def enqueue(self, request_data: Any) -> Any:
        future = asyncio.Future()
        await self.queue.put((request_data, future))
        return await future
