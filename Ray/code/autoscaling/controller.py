import math
import time
import os
from typing import Dict, Any, List, Optional

import psutil
import ray

from .metrics import TokenMetrics
from .predictor import AsymmetricEMA
from .capacity import OnlineCapacityModel
from .scaling_policy import AdvancedScalingPolicy, AdvancedScalingPolicyConfig
import config


@ray.remote(num_cpus=0)
class AutoscalingController:
    """
    Orchestrates metrics collection, demand prediction, capacity calibration,
    and time-driven scaling decisions.
    """

    def __init__(
        self,
        window_size_s: int = 60,
        policy_config: AdvancedScalingPolicyConfig = None,
    ):
        self.metrics = TokenMetrics(window_size_s=window_size_s)
        self.predictor_prefill = AsymmetricEMA(alpha_up=0.7, alpha_down=0.3)
        self.predictor_decode = AsymmetricEMA(alpha_up=0.7, alpha_down=0.3)
        self.capacity = OnlineCapacityModel()
        self.policy = AdvancedScalingPolicy(policy_config)
        self._last_decision: Dict[str, Any] = {}
        self._scaling_log: List[Dict[str, Any]] = []

    def record(self, input_tokens: int, output_tokens: int, latency_ms: float,
               cpu_util: Optional[float] = None, rss_bytes: Optional[int] = None):
        self.metrics.record_end(input_tokens, output_tokens, latency_ms)
        self.capacity.update(input_tokens, output_tokens, latency_ms / 1000.0)

        if cpu_util is not None and rss_bytes is not None:
            self.metrics.record_replica_stats(cpu_util, rss_bytes)

    def record_start(self):
        self.metrics.record_start()

    def record_probe(self, input_tokens: int, output_tokens: int, latency_ms: float):
        self.capacity.seed_probe(input_tokens, output_tokens, latency_ms / 1000.0)

    def decide(self, current_replicas: int) -> Dict[str, Any]:
        prefill_tps = self.metrics.prefill_tokens_per_sec()
        decode_tps = self.metrics.decode_tokens_per_sec()

        pred_prefill = self.predictor_prefill.update(prefill_tps)
        pred_decode = self.predictor_decode.update(decode_tps)

        latency_stats = self.metrics.latency_stats()
        inflight = self.metrics.get_inflight_requests()
        replica_cpu = self.metrics.replica_cpu_util()
        observed_rss = self.metrics.observed_rss_bytes()

        static_ceiling = config.static_max_replicas()
        if observed_rss > 0:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            ram_ceiling = max(1, math.floor(total_ram_gb * 0.65 / (observed_rss / (1024**3))))
            dynamic_ceiling = min(static_ceiling, ram_ceiling)
        else:
            dynamic_ceiling = static_ceiling
        dynamic_ceiling = max(1, dynamic_ceiling)

        recommended_replicas = self.policy.compute_replicas(
            pred_prefill_tps=pred_prefill,
            pred_decode_tps=pred_decode,
            prefill_rate=self.capacity.prefill_rate(),
            decode_rate=self.capacity.decode_rate(),
            inflight_requests=inflight,
            latency_stats=latency_stats,
            replica_cpu_util=replica_cpu,
            replica_ceiling=dynamic_ceiling,
        )

        signals = {
            "pred_prefill_tps": pred_prefill,
            "pred_decode_tps": pred_decode,
            "prefill_rate": self.capacity.prefill_rate(),
            "decode_rate": self.capacity.decode_rate(),
            "calibrated": self.capacity.is_calibrated(),
            "inflight": inflight,
            "p95_ms": latency_stats.get("p95", 0.0),
            "replica_cpu_util": replica_cpu,
            "dynamic_ceiling": dynamic_ceiling,
        }

        decision = {
            "from": current_replicas,
            "to": recommended_replicas,
            "signals": signals,
            "ts": time.time(),
        }

        if recommended_replicas != current_replicas:
            self._scaling_log.append(decision)

        self._last_decision = decision
        return decision

    def get_latest_decision(self) -> Dict[str, Any]:
        return self._last_decision

    def get_scaling_log(self) -> List[Dict[str, Any]]:
        return list(self._scaling_log)

    def get_capacity_stats(self) -> Dict:
        return self.capacity.stats()
