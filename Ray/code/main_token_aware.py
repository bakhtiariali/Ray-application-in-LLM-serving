import json
import os
import signal
import sys
import time
from datetime import datetime

import httpx
import ray
from ray import serve

from llm_deployment import BaseLLMDeployment
from autoscaling.controller import AutoscalingController
import config

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class TokenAwareDeployment(BaseLLMDeployment):
    pass


def send_probe(client, prompt_tokens, max_new_tokens=100):
    payload = {"message": "x" * prompt_tokens}
    start = time.time()
    try:
        resp = client.post("http://localhost:8000/chat", json=payload,
                           timeout=httpx.Timeout(180.0))
        latency_ms = (time.time() - start) * 1000.0
        if resp.status_code == 200:
            data = resp.json()
            metrics = data.get("metrics", {})
            return metrics.get("input_tokens", 0), metrics.get("output_tokens", 0), latency_ms
    except Exception:
        pass
    return 0, 0, 0


def control_loop(controller, deployment_class, scaling_log_path):
    current_replicas = 1

    while True:
        try:
            decision = ray.get(controller.decide.remote(current_replicas))
            desired = decision["to"]
            signals = decision["signals"]

            print(f"[AUTOSCALER] tick | replicas={current_replicas} desired={desired} | "
                  f"prefill={signals['pred_prefill_tps']:.2f} decode={signals['pred_decode_tps']:.2f} "
                  f"inflight={signals['inflight']} p95={signals['p95_ms']:.1f}ms "
                  f"rate_p={signals['prefill_rate']:.1f} rate_d={signals['decode_rate']:.1f} "
                  f"calibrated={signals['calibrated']} ceiling={signals['dynamic_ceiling']}", flush=True)

            if desired != current_replicas:
                print(f"[AUTOSCALER] Scaling {current_replicas} -> {desired}", flush=True)
                serve.run(
                    deployment_class.options(num_replicas=desired).bind(
                        controller_handle=controller
                    ),
                    route_prefix="/chat",
                    blocking=False,
                )
                current_replicas = desired
        except ray.exceptions.RayActorError:
            print("[AUTOSCALER] Controller actor died, continuing...", flush=True)

        time.sleep(config.CONTROL_INTERVAL_S)


def _export_scaling_log(controller, scaling_log_path):
    try:
        scaling_log = ray.get(controller.get_scaling_log.remote())
        cap_stats = ray.get(controller.get_capacity_stats.remote())
    except Exception:
        return
    os.makedirs("results", exist_ok=True)
    with open(scaling_log_path, "w") as f:
        json.dump({"scaling_log": scaling_log, "capacity": cap_stats}, f, indent=2, default=str)
    print(f"\n[AUTOSCALER] Scaling log saved to {scaling_log_path}", flush=True)


def _signal_handler(signum, frame):
    raise KeyboardInterrupt()


if __name__ == "__main__":
    ray.init()
    serve.start()

    controller = AutoscalingController.remote()

    app = TokenAwareDeployment.bind(controller_handle=controller)
    serve.run(app, route_prefix="/chat")

    print("Server running at http://localhost:8000/chat", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaling_log_path = f"results/scaling_log_{ts}.json"

    print("[AUTOSCALER] Sending startup probes...", flush=True)
    time.sleep(5)
    with httpx.Client() as client:
        for prompt_tokens in [20, 120, 300]:
            inp, out, lat = send_probe(client, prompt_tokens)
            if inp > 0 and out > 0:
                controller.record_probe.remote(inp, out, lat)
                print(f"[AUTOSCALER] Probe: {inp}/{out} tokens, {lat:.0f}ms", flush=True)
            else:
                print(f"[AUTOSCALER] Probe failed for {prompt_tokens} tokens, continuing...", flush=True)
        cap = ray.get(controller.get_capacity_stats.remote())
        print(f"[AUTOSCALER] Capacity after probes: {cap}", flush=True)

    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        control_loop(controller, TokenAwareDeployment, scaling_log_path)
    except KeyboardInterrupt:
        print("\n[AUTOSCALER] Shutting down...", flush=True)
        _export_scaling_log(controller, scaling_log_path)