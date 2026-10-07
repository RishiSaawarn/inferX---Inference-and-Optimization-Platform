import time
import logging
import threading
from typing import Callable, Any

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0
        
        # P2-13: HALF_OPEN probing counters and thread safety
        self._lock = threading.Lock()
        self.probes_in_flight = 0
        self.max_probes = 3
        self.success_count = 0
        self.success_threshold = 2

    def record_failure(self):
        with self._lock:
            if self.state == "HALF_OPEN":
                # Any failure in HALF_OPEN trips it back to OPEN immediately
                self.state = "OPEN"
                self.last_failure_time = time.monotonic()
                self.probes_in_flight = max(0, self.probes_in_flight - 1)
                logger.warning("Circuit Breaker probe failed, tripped back to OPEN")
                return
                
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                self.last_failure_time = time.monotonic()
                logger.warning(f"Circuit Breaker tripped OPEN after {self.failures} consecutive failures")

    def record_success(self):
        with self._lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                self.probes_in_flight = max(0, self.probes_in_flight - 1)
                
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.failures = 0
                    self.success_count = 0
                    logger.info("Circuit Breaker recovered, state is CLOSED")
            elif self.state == "CLOSED":
                self.failures = 0

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
                
            if self.state == "OPEN":
                # Check if recovery timeout has passed
                now = time.monotonic()
                if (now - self.last_failure_time) > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.probes_in_flight = 0
                    self.success_count = 0
                    logger.info("Circuit Breaker entering HALF_OPEN state")
                else:
                    return False
                    
            if self.state == "HALF_OPEN":
                # Let only max_probes through to test the waters
                if self.probes_in_flight < self.max_probes:
                    self.probes_in_flight += 1
                    return True
                else:
                    return False
                    
            return False
