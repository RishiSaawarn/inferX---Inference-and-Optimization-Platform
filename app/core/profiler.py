import time


class Profiler:
    def __init__(self):
        self.timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}
        self.start_time = time.perf_counter()

    def start(self, segment: str):
        self._starts[segment] = time.perf_counter()

    def stop(self, segment: str):
        if segment in self._starts:
            elapsed = time.perf_counter() - self._starts[segment]
            self.timings[segment] = self.timings.get(segment, 0.0) + elapsed * 1000.0
            del self._starts[segment]

    def get_results(self) -> dict[str, float]:
        total = (time.perf_counter() - self.start_time) * 1000.0
        results = {"request_total_ms": round(total, 2)}
        for k, v in self.timings.items():
            results[f"{k}_ms"] = round(v, 2)
        return results
