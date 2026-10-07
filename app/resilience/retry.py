import asyncio
import logging
import random
import inspect
from typing import Callable, Any

logger = logging.getLogger(__name__)

# P2-12: List of exceptions safe to retry
TRANSIENT_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    # Add other network/transient exceptions
)

async def with_retry(func: Callable, max_retries: int = 3, base_delay: float = 0.5) -> Any:
    for attempt in range(max_retries + 1):
        try:
            if inspect.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except TRANSIENT_EXCEPTIONS as e:
            if attempt == max_retries:
                logger.error(f"Action failed after {max_retries} retries: {e}")
                raise
                
            # P2-12: Full Jitter
            max_delay = base_delay * (2 ** attempt)
            delay = random.uniform(0, max_delay)
            logger.warning(f"Transient error: {e}. Retrying {attempt+1}/{max_retries} after {delay:.2f}s")
            await asyncio.sleep(delay)
        except Exception as e:
            # Non-transient exceptions bubble up immediately
            logger.error(f"Non-transient error, failing immediately: {e}")
            raise
