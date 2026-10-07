import time
from typing import Dict
from contextlib import contextmanager

class RequestProfiler:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        self.start_times[name] = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.timings[name] = (end - self.start_times[name]) * 1000.0

    def get_summary(self) -> Dict[str, float]:
        return self.timings.copy()
