import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from transformers import AutoTokenizer


# ---------------- CONFIG ----------------

URL = "http://localhost:8000/chat"
CONCURRENCY = 4

ARENA_PROMPT_FILE = "workloads/arena_prompt_set_v1.json"

# Use the SAME tokenizer as your deployed TinyLlama model
TOKENIZER_MODEL_NAME = "./models/tinyllama"

SHORT_MAX_TOKENS = 64
MEDIUM_MAX_TOKENS = 256
MAX_SAMPLES_PER_BUCKET = 10

OUTPUT_FILE = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ----------------------------------------


def load_prompts(path: str) -> List[Dict[str, Any]]:
    """
    Load prompts from arena_prompt_set_v1.json.

    Supports:
    1. {"prompts": [...]}
    2. [...]
    """
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


async def call_model(client: httpx.AsyncClient, prompt: str) -> Optional[str]:
    """
    Send one prompt to your Ray Serve TinyLlama endpoint.
    """
    payload = {
        "message": prompt
    }

    try:
        response = await client.post(
            URL,
            json=payload,
            timeout=httpx.Timeout(120.0),
        )

        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {response.text}")
            return None

        data = response.json()

        if "response" not in data:
            print(f"Unexpected response format: {data}")
            return None

        return data["response"]

    except Exception as e:
        print(f"Request failed: {e}")
        return None


async def main():
    print(f"Loading tokenizer from: {TOKENIZER_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL_NAME,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading prompt set from: {ARENA_PROMPT_FILE}")
    prompts = load_prompts(ARENA_PROMPT_FILE)

    buckets = {
        "short": [],
        "medium": [],
        "long": [],
    }

    # Prepare prompts
    for p in prompts:
        prompt_text = p.get("prompt")

        if not prompt_text:
            continue

        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)

        # Recalculate token count using TinyLlama tokenizer
        # This is more accurate than using stored GPT-2 token counts.
        input_tokens = len(
            tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
            )
        )

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

    # Select samples from each bucket
    selected = []

    for bucket_name in ["short", "medium", "long"]:
        bucket_items = buckets[bucket_name]
        sample_count = min(MAX_SAMPLES_PER_BUCKET, len(bucket_items))

        if sample_count == 0:
            print(f"Warning: no prompts found for bucket: {bucket_name}")
            continue

        selected.extend(
            random.sample(bucket_items, sample_count)
        )

    random.shuffle(selected)

    print(f"Selected {len(selected)} prompts total.")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Endpoint: {URL}")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    limits = httpx.Limits(
        max_connections=CONCURRENCY,
        max_keepalive_connections=CONCURRENCY,
    )

    results: List[Dict[str, Any]] = []

    total_start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits) as client:

        async def worker(item: Dict[str, Any], idx: int):
            async with semaphore:
                start = time.perf_counter()

                output_text = await call_model(
                    client=client,
                    prompt=item["prompt"],
                )

                latency = time.perf_counter() - start

                if output_text is None:
                    return

                output_tokens = len(
                    tokenizer.encode(
                        output_text,
                        add_special_tokens=False,
                    )
                )

                tokens_per_second = (
                    output_tokens / latency if latency > 0 else 0.0
                )

                results.append(
                    {
                        **item,
                        "prompt_index": idx,
                        "output_tokens": output_tokens,
                        "latency_sec": latency,
                        "tokens_per_second": tokens_per_second,
                        "response": output_text,
                    }
                )

        tasks = [
            worker(item, i)
            for i, item in enumerate(selected)
        ]

        await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - total_start

    results.sort(key=lambda x: x["prompt_index"])

    successful_requests = len(results)
    failed_requests = len(selected) - successful_requests

    throughput_rps = (
        successful_requests / total_duration
        if total_duration > 0
        else 0.0
    )

    print("\nBenchmark finished.")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Total duration: {total_duration:.2f} sec")
    print(f"Throughput: {throughput_rps:.4f} requests/sec")

    # Basic summary metrics
    latencies = [r["latency_sec"] for r in results]
    output_tokens_list = [r["output_tokens"] for r in results]
    tokens_per_second_list = [r["tokens_per_second"] for r in results]

    summary = {
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "total_duration_sec": total_duration,
        "throughput_rps": throughput_rps,
    }

    if results:
        summary.update(
            {
                "avg_latency_sec": sum(latencies) / len(latencies),
                "min_latency_sec": min(latencies),
                "max_latency_sec": max(latencies),
                "avg_output_tokens": sum(output_tokens_list) / len(output_tokens_list),
                "avg_tokens_per_second": sum(tokens_per_second_list)
                / len(tokens_per_second_list),
            }
        )

    output_data = {
        "config": {
            "url": URL,
            "prompt_file": ARENA_PROMPT_FILE,
            "tokenizer_model": TOKENIZER_MODEL_NAME,
            "short_max_tokens": SHORT_MAX_TOKENS,
            "medium_max_tokens": MEDIUM_MAX_TOKENS,
            "max_samples_per_bucket": MAX_SAMPLES_PER_BUCKET,
            "concurrency": CONCURRENCY,
        },
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
