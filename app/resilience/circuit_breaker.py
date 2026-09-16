import time
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0

    def record_failure(self):
        self.failures += 1
        if self.state == CircuitState.CLOSED and self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning("Circuit breaker OPEN")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning("Circuit breaker OPEN (failed in HALF_OPEN)")

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failures = 0
            logger.info("Circuit breaker CLOSED (recovered)")
        elif self.state == CircuitState.CLOSED:
            self.failures = 0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker HALF_OPEN (testing recovery)")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            # allow one request through
            return True
        return False
