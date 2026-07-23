import argparse
import glob
import json
import os
from datetime import datetime


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def replica_seconds_from_timeline(path):
    if not path or not os.path.exists(path):
        return None
    data = load_json(path)
    if not data:
        return 0.0
    total = 0.0
    for i in range(1, len(data)):
        dt = data[i]["ts"] - data[i - 1]["ts"]
        replicas = sum(data[i].get("replica_states", {}).values())
        total += replicas * dt
    return total


def replica_seconds_from_scaling_log(path):
    if not path or not os.path.exists(path):
        return None
    data = load_json(path)
    log = data.get("scaling_log", [])
    if not log:
        return 0.0
    total = 0.0
    for i, entry in enumerate(log):
        dt = 0.0
        if i < len(log) - 1:
            dt = log[i + 1]["ts"] - entry["ts"]
        else:
            dt = 60.0
        total += entry["to"] * max(dt, 0)
    return total


def plot_timelines(baseline_path, tokenaware_path, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not available, skipping plot.")
        return

    baseline_data = load_json(baseline_path) if baseline_path and os.path.exists(baseline_path) else []
    tokenaware_data = load_json(tokenaware_path) if tokenaware_path and os.path.exists(tokenaware_path) else []

    fig, ax = plt.subplots(figsize=(12, 5))

    if baseline_data:
        t0 = baseline_data[0]["ts"]
        times_b = [d["ts"] - t0 for d in baseline_data]
        reps_b = [sum(d.get("replica_states", {}).values()) for d in baseline_data]
        ax.step(times_b, reps_b, where="post", label="Ray built-in autoscaler", linewidth=2)

    if tokenaware_data:
        t0 = tokenaware_data[0]["ts"]
        times_t = [d["ts"] - t0 for d in tokenaware_data]
        reps_t = [sum(d.get("replica_states", {}).values()) for d in tokenaware_data]
        ax.step(times_t, reps_t, where="post", label="Token-aware autoscaler", linewidth=2)

    ax.set_xlabel("Elapsed time (seconds)")
    ax.set_ylabel("Running replicas")
    ax.set_title("Replica Count Over Time: Baseline vs Token-Aware")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nReplica timeline plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--baseline-timeline", default=None)
    parser.add_argument("--tokenaware-timeline", default=None)
    args = parser.parse_args()

    a = load_json(args.file_a)
    b = load_json(args.file_b)

    print("=" * 80)
    print(f"{'Metric':<35} {'Baseline':<20} {'Token-Aware':<20}")
    print("=" * 80)

    for metric, key_a, key_b, fmt in [
        ("Successful / Failed", ("successful_requests", "failed_requests"), ("successful_requests", "failed_requests"), "combined"),
        ("Throughput (req/s)", "throughput_rps", "throughput_rps", ".4f"),
        ("Avg Latency (s)", "avg_latency_sec", "avg_latency_sec", ".2f"),
        ("P50 Latency (s)", "p50_latency_sec", "p50_latency_sec", ".2f"),
        ("P95 Latency (s)", "p95_latency_sec", "p95_latency_sec", ".2f"),
        ("SLO Attainment (%)", "slo_attainment_rate", "slo_attainment_rate", ".1%"),
        ("Goodput (out tok/s)", "goodput_output_tokens_per_sec", "goodput_output_tokens_per_sec", ".2f"),
        ("Failure Rate", "failure_rate", "failure_rate", ".2%"),
    ]:
        sa = a["summary"]
        sb = b["summary"]
        if fmt == "combined":
            va = f"{sa.get(key_a[0], '?')} / {sa.get(key_a[1], '?')}"
            vb = f"{sb.get(key_b[0], '?')} / {sb.get(key_b[1], '?')}"
        else:
            va = format(sa.get(key_a, 0), fmt) if isinstance(sa.get(key_a), (int, float)) else str(sa.get(key_a, "?"))
            vb = format(sb.get(key_b, 0), fmt) if isinstance(sb.get(key_b), (int, float)) else str(sb.get(key_b, "?"))
        print(f"{metric:<35} {va:<20} {vb:<20}")

    rep_secs_a = replica_seconds_from_timeline(args.baseline_timeline)
    rep_secs_b = replica_seconds_from_timeline(args.tokenaware_timeline)
    if rep_secs_b is None:
        scaling_log_path = f"results/scaling_log_{datetime.now().strftime('%Y%m%d')}*.json"
        logs = sorted(glob.glob(scaling_log_path), reverse=True)
        if logs:
            rep_secs_b = replica_seconds_from_scaling_log(logs[0])

    va = f"{rep_secs_a:.0f}" if rep_secs_a is not None else "N/A"
    vb = f"{rep_secs_b:.0f}" if rep_secs_b is not None else "N/A"
    print(f"{'Replica-Seconds':<35} {va:<20} {vb:<20}")

    print("=" * 80)

    slo_a = a["summary"].get("slo_attainment_rate", 0)
    slo_b = b["summary"].get("slo_attainment_rate", 0)
    goodput_a = a["summary"].get("goodput_output_tokens_per_sec", 0)
    goodput_b = b["summary"].get("goodput_output_tokens_per_sec", 0)

    print("\nComparison: ", end="")
    if slo_b > slo_a and goodput_b > goodput_a:
        print("TOKEN-AWARE WINS (higher SLO attainment + goodput)")
    elif slo_b > slo_a * 1.05:
        print("TOKEN-AWARE BETTER SLO attainment")
    elif goodput_b > goodput_a * 1.05:
        print("TOKEN-AWARE BETTER goodput")
    elif abs(slo_b - slo_a) < 0.05 and abs(goodput_b - goodput_a) < 0.5:
        print("TIE (within noise)")
    else:
        print("BASELINE faster or comparable")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/replica_timeline_comparison_{ts}.png"
    plot_timelines(args.baseline_timeline, args.tokenaware_timeline, plot_path)


if __name__ == "__main__":
    main()