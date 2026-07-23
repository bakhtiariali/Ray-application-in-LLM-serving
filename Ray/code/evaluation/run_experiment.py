import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import requests


SERVER_URL = "http://localhost:8000/chat"


def wait_for_server(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.post(SERVER_URL, json={"message": "ping"}, timeout=30)
            if r.status_code == 200:
                print("Server is ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("Server did not start within timeout.")
    return False


def run_server(script):
    print(f"\nStarting server: {script}")
    return subprocess.Popen([sys.executable, script])


def stop_server(server):
    print("Stopping server...")
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("Force killing server...")
        server.kill()
    time.sleep(5)


def run_benchmark(profile="step", tag="", output=""):
    cmd = [sys.executable, "benchmark_arena.py", "--seed", "42", "--profile", profile]
    if tag:
        cmd += ["--tag", tag]
    if output:
        cmd += ["--output", output]
    print(f"\nRunning benchmark: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def replica_timeline_poller(stop_event, output_path):
    import ray
    ray.init(address="auto", ignore_reinit_error=True)
    from ray import serve

    timeline = []
    while not stop_event.is_set():
        try:
            status = serve.status()
            replica_states = {}
            for app_name, app_status in status.applications.items():
                for dep_name, dep_status in app_status.deployments.items():
                    running = len(dep_status.replica_states.get("RUNNING", []))
                    replica_states[dep_name] = running
            timeline.append({"ts": time.time(), "replica_states": replica_states})
        except Exception as e:
            print(f"[poller] Warning: {e}")
        time.sleep(1)

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2, default=str)
    print(f"Replica timeline saved to {output_path}")


if __name__ == "__main__":
    print("Cleaning previous Ray processes...")
    subprocess.run(["ray", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_result = f"results/baseline_{ts}.json"
    tokenaware_result = f"results/tokenaware_{ts}.json"
    baseline_timeline = f"results/replica_timeline_baseline_{ts}.json"
    tokenaware_timeline = f"results/replica_timeline_tokenaware_{ts}.json"

    # Baseline experiment
    server = run_server("main_baseline.py")
    if not wait_for_server():
        stop_server(server)
        sys.exit("Baseline server failed to start.")

    stop_event = threading.Event()
    poller_thread = threading.Thread(target=replica_timeline_poller, args=(stop_event, baseline_timeline), daemon=True)
    poller_thread.start()

    run_benchmark(profile="step", tag="baseline", output=baseline_result)

    stop_event.set()
    poller_thread.join(timeout=10)
    stop_server(server)
    subprocess.run(["ray", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Token-aware experiment
    server = run_server("main_token_aware.py")
    if not wait_for_server():
        stop_server(server)
        sys.exit("Token-aware server failed to start.")

    stop_event = threading.Event()
    poller_thread = threading.Thread(target=replica_timeline_poller, args=(stop_event, tokenaware_timeline), daemon=True)
    poller_thread.start()

    run_benchmark(profile="step", tag="tokenaware", output=tokenaware_result)

    stop_event.set()
    poller_thread.join(timeout=10)
    stop_server(server)
    subprocess.run(["ray", "stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("\nExperiments completed.")

    # Run comparison
    print("\nComparing results...")
    subprocess.run([sys.executable, "evaluation/compare_results.py",
                    baseline_result, tokenaware_result,
                    "--baseline-timeline", baseline_timeline,
                    "--tokenaware-timeline", tokenaware_timeline])
