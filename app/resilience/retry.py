import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

async def execute_with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    **kwargs
) -> Any:
    import random
    
    attempts = 0
    while attempts <= max_retries:
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            attempts += 1
            if attempts > max_retries:
                logger.error(f"Failed after {max_retries} retries: {e}")
                raise
            
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** (attempts - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            sleep_time = delay + jitter
            
            logger.warning(f"Attempt {attempts} failed: {e}. Retrying in {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)
