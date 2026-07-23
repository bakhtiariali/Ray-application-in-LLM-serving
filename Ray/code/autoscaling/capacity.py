from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np


class OnlineCapacityModel:
    """
    Learns per-replica service-time model via online linear regression:
        latency_s ≈ a·input_tokens + b·output_tokens + c

    Rates: prefill_rate = 1/a, decode_rate = 1/b (tokens/s per replica).
    Static fallback constants used until calibrated.
    """

    def __init__(self, windows: int = 200, refit_every: int = 5):
        self._samples: Deque[Tuple[float, float, float]] = deque(maxlen=windows)
        self._refit_every = refit_every
        self._sample_count_since_refit = 0

        self._a: float = 0.0
        self._b: float = 0.0
        self._c: float = 0.0
        self._calibrated: bool = False

        self._fallback_prefill_rate: float = 50.0
        self._fallback_decode_rate: float = 8.0

    def update(self, input_tokens: int, output_tokens: int, latency_s: float):
        self._samples.append((float(input_tokens), float(output_tokens), latency_s))
        self._sample_count_since_refit += 1

        if self._sample_count_since_refit >= self._refit_every and len(self._samples) >= 8:
            self._refit()
            self._sample_count_since_refit = 0

    def seed_probe(self, input_tokens: int, output_tokens: int, latency_s: float):
        self.update(input_tokens, output_tokens, latency_s)

    def _refit(self):
        n = len(self._samples)
        X = np.zeros((n, 3))
        y = np.zeros(n)

        for i, (inp, out, lat) in enumerate(self._samples):
            X[i, 0] = inp
            X[i, 1] = out
            X[i, 2] = 1.0
            y[i] = lat

        ridge_diag = np.eye(3) * 1e-3
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X.T @ X + ridge_diag, X.T @ y, rcond=None)
        except np.linalg.LinAlgError:
            return

        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        a = max(a, 1e-4)
        b = max(b, 1e-4)

        self._a = a
        self._b = b
        self._c = c

        if n >= 8 and self._a > 1e-4 and self._b > 1e-4:
            self._calibrated = True

    def is_calibrated(self) -> bool:
        return self._calibrated

    def prefill_rate(self) -> float:
        if not self._calibrated:
            return self._fallback_prefill_rate
        return 1.0 / self._a if self._a > 0 else self._fallback_prefill_rate

    def decode_rate(self) -> float:
        if not self._calibrated:
            return self._fallback_decode_rate
        return 1.0 / self._b if self._b > 0 else self._fallback_decode_rate

    def service_time(self, input_tokens: int, output_tokens: int) -> float:
        if not self._calibrated:
            return input_tokens / self._fallback_prefill_rate + output_tokens / self._fallback_decode_rate
        return self._a * input_tokens + self._b * output_tokens + self._c

    def stats(self) -> Dict:
        return {
            "calibrated": self._calibrated,
            "samples": len(self._samples),
            "a": self._a,
            "b": self._b,
            "c": self._c,
            "prefill_rate": self.prefill_rate(),
            "decode_rate": self.decode_rate(),
        }
