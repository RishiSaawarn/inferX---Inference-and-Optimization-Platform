import json
import hashlib
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
from app.core.config import settings
from app.observability.metrics import record_cache

logger = logging.getLogger(__name__)

class ResponseCache:
    def __init__(self):
        self.url = settings.app_config.redis.url if settings.app_config else "redis://localhost:6379"
        self.ttl = settings.app_config.redis.ttl if settings.app_config else 3600
        # Fail OPEN - if Redis isn't there, we just proceed without cache
        try:
            self.redis = redis.from_url(self.url, decode_responses=True)
            logger.info(f"Initialized Redis cache at {self.url}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis, caching disabled: {e}")
            self.redis = None

    def _generate_key(self, model_version: str, input_data: Any) -> str:
        # Convert input data to string conceptually, then hash
        input_str = json.dumps(input_data, sort_keys=True)
        key_raw = f"{model_version}_{input_str}".encode('utf-8')
        return hashlib.sha256(key_raw).hexdigest()

    async def get(self, model_version: str, input_data: Any) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            key = self._generate_key(model_version, input_data)
            val = await self.redis.get(key)
            if val:
                record_cache(model_version.split(':')[0], hit=True)
                return json.loads(val)
            record_cache(model_version.split(':')[0], hit=False)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    async def set(self, model_version: str, input_data: Any, response: Dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            key = self._generate_key(model_version, input_data)
            await self.redis.set(key, json.dumps(response), ex=self.ttl)
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
