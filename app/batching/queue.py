import asyncio
import logging
import time
from typing import Any, Callable, List, Tuple
from app.core.exceptions import QueueFull

logger = logging.getLogger(__name__)

class BatchQueue:
    def __init__(self, processor_callback: Callable, max_batch_size: int = 32, max_wait_ms: int = 15, max_queue_size: int = 1000):
        self.processor_callback = processor_callback
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        # P2-6: Back pressure
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.task: asyncio.Task | None = None
        self._running = False
        
    def start(self):
        if not self._running:
            self._running = True
            loop = asyncio.get_running_loop()
            self.task = loop.create_task(self._process_queue())
            
    async def stop(self):
        self._running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # P2-5: Draining the queue on shutdown
        while not self.queue.empty():
            item = self.queue.get_nowait()
            _, _, future = item
            if not future.done():
                future.set_exception(RuntimeError("Shutting down"))

    async def _process_queue(self):
        while self._running:
            try:
                # P2-3: Don't wait for max_wait if nothing is there.
                # Just wait indefinitely for the FIRST item of the new batch.
                first_item = await self.queue.get()
                
                batch = [first_item]
                batch_size = 1
                
                first_req_data, first_enqueue_time, first_future = first_item
                
                # P2-2: Compute deadline based on when the request was ENQUEUED, not now.
                deadline = first_enqueue_time + self.max_wait_ms
                
                while batch_size < self.max_batch_size:
                    now = time.monotonic()
                    timeout = deadline - now
                    
                    if timeout <= 0:
                        break
                        
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(item)
                        batch_size += 1
                    except asyncio.TimeoutError:
                        break
                        
                if batch:
                    await self._execute_batch(batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch queue loop: {e}", exc_info=True)

    async def _execute_batch(self, batch: List[Tuple[Any, float, asyncio.Future]]):
        requests = []
        futures = []
        row_counts = []
        
        for req, enqueue_time, future in batch:
            requests.append(req)
            futures.append(future)
            # Find number of rows per request for splitting later (P2-1)
            # Fallback to 1 if we can't determine it (like for unbatched nested lists)
            length = len(req) if isinstance(req, list) else 1
            # If it's a tensor
            if hasattr(req, "shape") and len(req.shape) > 0:
                length = req.shape[0]
            row_counts.append(length)
            
        try:
            # The processor concatenates everything.
            outputs = await self.processor_callback(requests)
            
            # Now we must partition the output back to futures (P2-1)
            # outputs might be a tensor.
            # Handle list outputs
            is_tensor = hasattr(outputs, "shape")
            
            offset = 0
            for r_cnt, future in zip(row_counts, futures):
                if not future.done():
                    if is_tensor and len(outputs.shape) > 0:
                        future.set_result(outputs[offset : offset + r_cnt])
                    elif isinstance(outputs, list):
                        future.set_result(outputs[offset : offset + r_cnt])
                    else:
                        # Fallback for scalar/atomic processing where it wasn't actually batched inside correctly
                        # or if batch size mapping failed. P2-4 safe fallback.
                        future.set_result(outputs)
                offset += r_cnt

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # P2-4: Handle malformed inputs by retrying individually if batch fails
            if len(batch) > 1:
                logger.info("Retrying batch individually due to failure.")
                for req, _, future in batch:
                    try:
                        res = await self.processor_callback([req])
                        if not future.done():
                            future.set_result(res[0] if isinstance(res, list) else res)
                    except Exception as individual_e:
                        if not future.done():
                            future.set_exception(individual_e)
            else:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    async def enqueue(self, request_data: Any) -> Any:
        # P2-2: Track enqueue time
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        try:
            # P2-6: Use put_nowait to immediately fail when queue is full
            self.queue.put_nowait((request_data, time.monotonic(), future))
        except asyncio.QueueFull:
            logger.warning("Batch queue is full, throwing QueueFull exception.")
            raise QueueFull("Service busy, too many requests.")
            
        return await future
