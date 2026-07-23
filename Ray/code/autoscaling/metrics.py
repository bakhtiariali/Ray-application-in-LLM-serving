import time
from collections import deque
from typing import Deque, Dict, Any, List, Optional

try:
    import psutil
except ImportError:
    psutil = None
    import warnings
    warnings.warn("psutil not installed; replica CPU/RSS stats unavailable", RuntimeWarning)


class TokenMetrics:
    """
    Tracks token demand (prefill/decode), latency, inflight requests,
    and optional per-replica CPU/RSS stats over a sliding window.
    Token counts are reported externally (no internal tokenization).
    """

    def __init__(self, window_size_s: int = 60):
        self.window_size_s = window_size_s

        self.history: Deque[Dict[str, Any]] = deque()
        self.replica_history: Deque[Dict[str, Any]] = deque()

        self.inflight_requests: int = 0

    def record_start(self):
        self.inflight_requests += 1

    def record_end(self, input_tokens: int, output_tokens: int, latency_ms: float):
        self.inflight_requests = max(0, self.inflight_requests - 1)

        ts = time.time()
        entry = {
            "ts": ts,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }
        self.history.append(entry)
        self._cleanup()

    def record_replica_stats(self, cpu_util: float, rss_bytes: int):
        if psutil is None:
            return
        self.replica_history.append({
            "ts": time.time(),
            "cpu_util": cpu_util,
            "rss_bytes": rss_bytes,
        })
        cutoff = time.time() - self.window_size_s
        while self.replica_history and self.replica_history[0]["ts"] < cutoff:
            self.replica_history.popleft()

    def _cleanup(self):
        cutoff = time.time() - self.window_size_s
        while self.history and self.history[0]["ts"] < cutoff:
            self.history.popleft()

    def token_demand_per_sec(self) -> float:
        self._cleanup()
        if not self.history:
            return 0.0
        total_tokens = sum(e["input_tokens"] + e["output_tokens"] for e in self.history)
        return total_tokens / float(self.window_size_s)

    def prefill_tokens_per_sec(self) -> float:
        self._cleanup()
        if not self.history:
            return 0.0
        total = sum(e["input_tokens"] for e in self.history)
        return total / float(self.window_size_s)

    def decode_tokens_per_sec(self) -> float:
        self._cleanup()
        if not self.history:
            return 0.0
        total = sum(e["output_tokens"] for e in self.history)
        return total / float(self.window_size_s)

    def latency_stats(self) -> Dict[str, float]:
        self._cleanup()
        if not self.history:
            return {"p50": 0.0, "p95": 0.0, "avg": 0.0}

        latencies = [e["latency_ms"] for e in self.history]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        def percentile(p: float) -> float:
            if n == 0:
                return 0.0
            k = int(p * (n - 1))
            return latencies_sorted[k]

        avg = sum(latencies) / n
        return {
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "avg": avg,
        }

    def get_inflight_requests(self) -> int:
        self._cleanup()
        return self.inflight_requests

    def replica_cpu_util(self) -> float:
        if psutil is None:
            return 0.0
        cutoff = time.time() - self.window_size_s
        values = [e["cpu_util"] for e in self.replica_history if e["ts"] >= cutoff]
        return max(values) if values else 0.0

    def observed_rss_bytes(self) -> int:
        if psutil is None:
            return 0
        cutoff = time.time() - self.window_size_s
        values = [e["rss_bytes"] for e in self.replica_history if e["ts"] >= cutoff]
        return max(values) if values else 0
