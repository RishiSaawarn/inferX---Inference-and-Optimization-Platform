import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

async def execute_with_fallback(primary_func: Callable, fallback_func: Callable, *args, **kwargs) -> Any:
    try:
        import asyncio
        if asyncio.iscoroutinefunction(primary_func):
            return await primary_func(*args, **kwargs)
        else:
            return primary_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Primary execution failed: {e}. Executing fallback...")
        import asyncio
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func(*args, **kwargs)
        else:
            return fallback_func(*args, **kwargs)
