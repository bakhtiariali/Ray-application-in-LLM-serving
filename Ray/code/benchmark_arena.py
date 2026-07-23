import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from transformers import AutoTokenizer


URL = "http://localhost:8000/chat"
ARENA_PROMPT_FILE = "workloads/arena_prompt_set_v1.json"
TOKENIZER_MODEL_NAME = "./models/tinyllama"
SHORT_MAX_TOKENS = 64
MEDIUM_MAX_TOKENS = 256


def load_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure in arena prompt file.")


def bucket_from_tokens(n: int) -> str:
    if n <= SHORT_MAX_TOKENS:
        return "short"
    if n <= MEDIUM_MAX_TOKENS:
        return "medium"
    return "long"


async def call_model(client: httpx.AsyncClient, prompt: str, timeout: float = 180.0) -> Optional[Dict[str, Any]]:
    payload = {"message": prompt}
    start = time.perf_counter()
    try:
        response = await client.post(URL, json=payload, timeout=httpx.Timeout(timeout))
        arrival_ts = time.perf_counter()
        latency = arrival_ts - start

        if response.status_code != 200:
            return {"response": "", "status": response.status_code, "error": response.text,
                    "http_status": response.status_code, "arrival_ts": arrival_ts, "latency_sec": latency}

        data = response.json()
        return {
            "response": data.get("response", ""),
            "status": response.status_code,
            "error": data.get("error"),
            "http_status": response.status_code,
            "arrival_ts": arrival_ts,
            "latency_sec": latency,
            "metrics": data.get("metrics", {}),
        }
    except Exception as e:
        return {"response": "", "status": -1, "error": repr(e),
                "http_status": -1, "arrival_ts": time.perf_counter(),
                "latency_sec": time.perf_counter() - start}


async def _run_wave(client, items: List[Dict], concurrency: int, tokenizer, timeout: float) -> List[Dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict] = []

    async def worker(item, idx):
        async with semaphore:
            result = await call_model(client, item["prompt"], timeout)

            if result is None or result.get("status", -1) != 200:
                return

            output_text = result.get("response", "")
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False)) if output_text else 0

            results.append({
                **item,
                "prompt_index": idx,
                "output_tokens": output_tokens,
                "latency_sec": result["latency_sec"],
                "arrival_ts": result["arrival_ts"],
                "http_status": result["http_status"],
                "error": result["error"],
                "response": output_text,
            })

    tasks = [worker(item, i) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)
    return results


PROFILES = {
    "fixed": None,
    "step": [(2, 6), (8, 16), (1, 4)],
    "spike": [(2, 4), (12, 12), (2, 4)],
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=["fixed", "step", "spike"], default="fixed")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--samples-per-bucket", type=int, default=10)
    parser.add_argument("--tag", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    url = args.url
    seed = args.seed
    concurrency = args.concurrency
    samples_per_bucket = args.samples_per_bucket
    timeout = args.timeout

    print(f"Loading tokenizer from: {TOKENIZER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading prompt set from: {ARENA_PROMPT_FILE}")
    prompts = load_prompts(ARENA_PROMPT_FILE)

    buckets = {"short": [], "medium": [], "long": []}
    for p in prompts:
        prompt_text = p.get("prompt")
        if not prompt_text:
            continue
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)
        input_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        bucket = bucket_from_tokens(input_tokens)
        item = {
            "id": p.get("id"),
            "question_id": p.get("question_id"),
            "turn": p.get("turn"),
            "language": p.get("language"),
            "bucket": bucket,
            "original_bucket": p.get("bucket"),
            "model_a": p.get("model_a"),
            "model_b": p.get("model_b"),
            "prompt": prompt_text,
            "input_tokens": input_tokens,
        }
        buckets[bucket].append(item)

    print("Bucket sizes after TinyLlama tokenization:")
    print({k: len(v) for k, v in buckets.items()})

    rng = random.Random(seed)
    selected = []
    for bucket_name in ["short", "medium", "long"]:
        bucket_items = buckets[bucket_name]
        sample_count = min(samples_per_bucket, len(bucket_items))
        if sample_count == 0:
            print(f"Warning: no prompts found for bucket: {bucket_name}")
            continue
        selected.extend(rng.sample(bucket_items, sample_count))
    rng.shuffle(selected)

    print(f"Selected {len(selected)} prompts total.")
    print(f"Profile: {args.profile}, Seed: {seed}")

    limits = httpx.Limits(max_connections=max(concurrency, 20), max_keepalive_connections=max(concurrency, 20))

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits) as client:
        profile_waves = PROFILES[args.profile]

        if profile_waves is None:
            results = await _run_wave(client, selected, concurrency, tokenizer, timeout)
        else:
            prompt_idx = 0
            for wave_conc, wave_n in profile_waves:
                wave_items = []
                for i in range(wave_n):
                    wave_items.append(selected[prompt_idx % len(selected)])
                    prompt_idx += 1
                wave_results = await _run_wave(client, wave_items, wave_conc, tokenizer, timeout)
                results.extend(wave_results)
                print(f"  Wave concurrency={wave_conc} n={wave_n}: {len(wave_results)}/ {wave_n} successful")

    total_duration = time.perf_counter() - total_start
    results.sort(key=lambda x: x.get("prompt_index", 0))

    successful_requests = len(results)
    failed_requests = len(selected) - successful_requests

    throughput_rps = successful_requests / total_duration if total_duration > 0 else 0.0

    print("\nBenchmark finished.")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Total duration: {total_duration:.2f} sec")
    print(f"Throughput: {throughput_rps:.4f} requests/sec")

    latencies = [r["latency_sec"] for r in results]
    output_tokens_list = [r["output_tokens"] for r in results]

    slo_threshold_sec = float(os.getenv("LLM_SLO_P95_MS", "30000")) / 1000.0
    slo_attainment = sum(1 for l in latencies if l < slo_threshold_sec) / len(latencies) if latencies else 0.0
    total_output_tokens = sum(output_tokens_list)
    goodput = total_output_tokens / total_duration if total_duration > 0 else 0.0

    summary = {
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "total_duration_sec": total_duration,
        "throughput_rps": throughput_rps,
        "failure_rate": failed_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0.0,
    }

    if results:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        def percentile(p):
            if n == 0:
                return 0.0
            k = int(p * (n - 1))
            return latencies_sorted[k]

        summary.update({
            "avg_latency_sec": sum(latencies) / n,
            "p50_latency_sec": percentile(0.5),
            "p95_latency_sec": percentile(0.95),
            "min_latency_sec": min(latencies),
            "max_latency_sec": max(latencies),
            "avg_output_tokens": sum(output_tokens_list) / n,
            "slo_attainment_rate": slo_attainment,
            "goodput_output_tokens_per_sec": goodput,
        })

    output_data = {
        "config": {
            "url": url,
            "prompt_file": ARENA_PROMPT_FILE,
            "tokenizer_model": TOKENIZER_MODEL_NAME,
            "short_max_tokens": SHORT_MAX_TOKENS,
            "medium_max_tokens": MEDIUM_MAX_TOKENS,
            "samples_per_bucket": samples_per_bucket,
            "concurrency": concurrency,
            "profile": args.profile,
            "seed": seed,
            "timeout": timeout,
        },
        "summary": summary,
        "results": results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag}" if args.tag else ""
    output_file = args.output or f"results/benchmark_results{tag_suffix}_{ts}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
