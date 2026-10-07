import pytest
import asyncio
from app.batching.queue import BatchQueue
from app.core.exceptions import QueueFull

@pytest.mark.asyncio
async def test_batch_execution():
    async def mock_processor(batch):
        return [1 for _ in batch]
        
    queue = BatchQueue(processor_callback=mock_processor, max_batch_size=2)
    queue.start()
    
    # Enqueue a few items
    f1 = await queue.enqueue([0, 1])
    f2 = await queue.enqueue([2, 3])
    
    res1, res2 = await asyncio.gather(f1, f2)
    assert res1 == 1
    assert res2 == 1
    
    await queue.stop()

@pytest.mark.asyncio
async def test_queue_full():
    queue = BatchQueue(processor_callback=lambda x: x, max_queue_size=1)
    
    t = asyncio.create_task(queue.enqueue(1))
    await asyncio.sleep(0.01) # let it process put_nowait
    
    with pytest.raises(QueueFull):
        await queue.enqueue(2)
        
    t.cancel()
