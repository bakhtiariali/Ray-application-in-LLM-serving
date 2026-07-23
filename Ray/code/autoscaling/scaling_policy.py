import math
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AdvancedScalingPolicyConfig:
    latency_slo_p95_ms: float = 30000.0
    safety_margin: float = 1.2
    target_requests_per_replica: float = 2.0
    scale_up_cooldown_s: float = 10.0
    scale_down_cooldown_s: float = 60.0
    downscale_hold_ticks: int = 3
    latency_boost_cap: float = 1.5
    min_replicas: int = 1
    high_cpu_util: float = 0.9


class AdvancedScalingPolicy:
    """
    Multi-signal autoscaling policy with phase-aware token demand,
    latency SLO boost, CPU boost, cooldowns, and downscale hysteresis.
    """

    def __init__(self, config: AdvancedScalingPolicyConfig = None):
        self.config = config or AdvancedScalingPolicyConfig()
        self._last_replicas = self.config.min_replicas
        self._last_scale_time = 0.0
        self._downscale_ticks = 0

    def _within_cooldown(self, now: float, increasing: bool) -> bool:
        cooldown = self.config.scale_up_cooldown_s if increasing else self.config.scale_down_cooldown_s
        return (now - self._last_scale_time) < cooldown

    def compute_replicas(
        self,
        pred_prefill_tps: float,
        pred_decode_tps: float,
        prefill_rate: float,
        decode_rate: float,
        inflight_requests: int,
        latency_stats: Dict[str, float],
        replica_cpu_util: float,
        replica_ceiling: int,
    ) -> int:
        now = time.time()
        cfg = self.config

        util = 0.0
        if prefill_rate > 0:
            util += pred_prefill_tps / prefill_rate
        if decode_rate > 0:
            util += pred_decode_tps / decode_rate
        token_replicas = math.ceil(util * cfg.safety_margin)

        if cfg.target_requests_per_replica > 0:
            queue_replicas = math.ceil(inflight_requests / cfg.target_requests_per_replica)
        else:
            queue_replicas = 0

        base = max(token_replicas, queue_replicas, cfg.min_replicas)

        p95 = latency_stats.get("p95", 0.0)
        if p95 > cfg.latency_slo_p95_ms and len(latency_stats) >= 5:
            if p95 > 0 and cfg.latency_slo_p95_ms > 0:
                factor = min(cfg.latency_boost_cap, p95 / cfg.latency_slo_p95_ms)
                base = max(base, math.ceil(base * factor))

        if replica_cpu_util >= cfg.high_cpu_util:
            base += 1

        desired = max(cfg.min_replicas, min(replica_ceiling, base))

        if desired > self._last_replicas:
            self._downscale_ticks = 0
            if self._within_cooldown(now, increasing=True):
                desired = self._last_replicas
            else:
                self._last_scale_time = now
                self._last_replicas = desired
        elif desired < self._last_replicas:
            self._downscale_ticks += 1
            if self._downscale_ticks < cfg.downscale_hold_ticks:
                desired = self._last_replicas
            elif self._within_cooldown(now, increasing=False):
                desired = self._last_replicas
            else:
                self._last_scale_time = now
                self._last_replicas = desired
                self._downscale_ticks = 0
        else:
            self._downscale_ticks = 0

        return desired
