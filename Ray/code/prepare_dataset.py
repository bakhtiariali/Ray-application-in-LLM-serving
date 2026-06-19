import os
import json
import random
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------- CONFIG ----------------

DATASET_NAME = "lmsys/chatbot_arena_conversations"
SPLIT = "train"

TOKENIZER_MODEL_NAME = "gpt2"

OUTPUT_DIR = "workloads"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arena_prompt_set_v1.json")

NUM_PER_BUCKET = 50

SHORT_MAX_TOKENS = 64
MEDIUM_MAX_TOKENS = 256

RANDOM_SEED = 42

# Set this to None if you want to scan the whole dataset.
# For faster testing you can use e.g. 10000.
MAX_ROWS_TO_SCAN = None

# ----------------------------------------


def is_english(sample):
    """
    Keep only English prompts.

    LMSYS dataset usually contains a 'language' field.
    We use it if available.
    """
    lang = sample.get("language", None)

    if lang is None:
        return False

    return str(lang).lower().strip() == "english"


def extract_prompt(sample):
    """
    Extract first user prompt from conversation_a.

    Expected structure:
    conversation_a = [
        {"role": "user", "content": "..."},
        ...
    ]
    """
    conv_a = sample.get("conversation_a", None)

    if not conv_a:
        return None

    if not isinstance(conv_a, list):
        return None

    first_turn = conv_a[0]

    if not isinstance(first_turn, dict):
        return None

    prompt = first_turn.get("content", None)

    if not prompt:
        return None

    if not isinstance(prompt, str):
        return None

    prompt = prompt.strip()

    if len(prompt) == 0:
        return None

    return prompt


def get_bucket(num_tokens):
    if num_tokens <= SHORT_MAX_TOKENS:
        return "short"
    elif num_tokens <= MEDIUM_MAX_TOKENS:
        return "medium"
    else:
        return "long"


def main():
    random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

    print(f"Loading dataset: {DATASET_NAME} / {SPLIT}")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    if MAX_ROWS_TO_SCAN is not None:
        max_rows = min(MAX_ROWS_TO_SCAN, len(ds))
        ds = ds.select(range(max_rows))

    print(f"Rows to scan: {len(ds)}")

    buckets = {
        "short": [],
        "medium": [],
        "long": [],
    }

    scanned = 0
    english_count = 0
    usable_count = 0

    for sample in ds:
        scanned += 1

        if not is_english(sample):
            continue

        english_count += 1

        prompt = extract_prompt(sample)
        if prompt is None:
            continue

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = len(input_ids)

        bucket = get_bucket(input_tokens)

        entry = {
            "id": f"{sample.get('question_id', 'unknown')}/turn{sample.get('turn', 0)}/a",
            "question_id": sample.get("question_id"),
            "turn": sample.get("turn"),
            "language": sample.get("language"),
            "bucket": bucket,
            "model_a": sample.get("model_a"),
            "model_b": sample.get("model_b"),
            "prompt": prompt,
            "input_tokens": input_tokens,
        }

        buckets[bucket].append(entry)
        usable_count += 1

    print("\n========== DATASET SCAN SUMMARY ==========")
    print(f"Scanned rows:     {scanned}")
    print(f"English rows:     {english_count}")
    print(f"Usable prompts:   {usable_count}")
    print(f"Short prompts:    {len(buckets['short'])}")
    print(f"Medium prompts:   {len(buckets['medium'])}")
    print(f"Long prompts:     {len(buckets['long'])}")
    print("==========================================\n")

    selected = []

    for bucket_name in ["short", "medium", "long"]:
        available = len(buckets[bucket_name])

        if available < NUM_PER_BUCKET:
            raise RuntimeError(
                f"Not enough {bucket_name} prompts. "
                f"Need {NUM_PER_BUCKET}, found {available}. "
                f"Try scanning more rows or changing thresholds."
            )

        sampled = random.sample(buckets[bucket_name], NUM_PER_BUCKET)
        selected.extend(sampled)

    # Sort for readability
    bucket_order = {"short": 0, "medium": 1, "long": 2}
    selected.sort(key=lambda x: (bucket_order[x["bucket"]], x["input_tokens"]))

    output = {
        "config": {
            "dataset": DATASET_NAME,
            "split": SPLIT,
            "tokenizer_model": TOKENIZER_MODEL_NAME,
            "num_per_bucket": NUM_PER_BUCKET,
            "short_max_tokens": SHORT_MAX_TOKENS,
            "medium_max_tokens": MEDIUM_MAX_TOKENS,
            "long_definition": f">{MEDIUM_MAX_TOKENS}",
            "language": "English",
            "random_seed": RANDOM_SEED,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "summary": {
            "total_prompts": len(selected),
            "short": NUM_PER_BUCKET,
            "medium": NUM_PER_BUCKET,
            "long": NUM_PER_BUCKET,
        },
        "prompts": selected,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved prompt set to: {OUTPUT_FILE}")
    print(f"Total saved prompts: {len(selected)}")


if __name__ == "__main__":
    main()





# # build_prompt_set.py
# import os
# import json
# import random
# from datetime import datetime

# from datasets import load_dataset
# from transformers import AutoTokenizer

# # ---------------- CONFIG ----------------
# TOKENIZER_MODEL_NAME = "gpt2"  # or your TinyLlama tokenizer if available
# MAX_SAMPLES = 5000             # upper bound on rows to inspect
# N_SHORT = 50
# N_MEDIUM = 50
# N_LONG = 50

# SHORT_MAX_TOKENS = 64          # <= this → short
# MEDIUM_MAX_TOKENS = 256        # (SHORT_MAX, MEDIUM_MAX] → medium
# OUTPUT_DIR = "workloads"
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arena_prompt_set_v1.json")
# # ----------------------------------------


# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print(f"Loading tokenizer: {TOKENIZER_MODEL_NAME}")
#     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)

#     print("Loading dataset: lmsys/chatbot_arena_conversations (train split)")
#     ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")

#     # Limit to a subset for speed (optional)
#     if len(ds) > MAX_SAMPLES:
#         indices = random.sample(range(len(ds)), MAX_SAMPLES)
#         ds = ds.select(indices)

#     print(f"Dataset subset size: {len(ds)}")

#     buckets = {
#         "short": [],
#         "medium": [],
#         "long": [],
#     }

#     inspected = 0
#     for sample in ds:
#         inspected += 1
#         # We use conversation_a[0]["content"] as our prompt
#         conv_a = sample.get("conversation_a", None)
#         if not conv_a or not isinstance(conv_a, list):
#             continue

#         first_turn = conv_a[0]
#         if not isinstance(first_turn, dict) or "content" not in first_turn:
#             continue

#         prompt = first_turn["content"]
#         if not isinstance(prompt, str) or not prompt.strip():
#             continue

#         # Tokenize
#         input_ids = tokenizer.encode(prompt, add_special_tokens=False)
#         num_tokens = len(input_ids)

#         # Decide bucket
#         if num_tokens <= SHORT_MAX_TOKENS:
#             bucket = "short"
#         elif num_tokens <= MEDIUM_MAX_TOKENS:
#             bucket = "medium"
#         else:
#             bucket = "long"

#         entry = {
#             "id": f"{sample.get('question_id', 'unknown')}/turn{sample.get('turn', 0)}/a",
#             "bucket": bucket,
#             "question_id": sample.get("question_id"),
#             "turn": sample.get("turn"),
#             "model_a": sample.get("model_a"),
#             "model_b": sample.get("model_b"),
#             "language": sample.get("language"),
#             "prompt": prompt,
#             "input_tokens": num_tokens,
#         }
#         buckets[bucket].append(entry)

#     print(f"Collected counts per bucket:")
#     for b in buckets:
#         print(f"  {b}: {len(buckets[b])}")

#     # Check that we have enough in each bucket
#     if len(buckets["short"]) < N_SHORT:
#         print(f"WARNING: only {len(buckets['short'])} short prompts, requested {N_SHORT}")
#         N_short_final = len(buckets["short"])
#     else:
#         N_short_final = N_SHORT

#     if len(buckets["medium"]) < N_MEDIUM:
#         print(f"WARNING: only {len(buckets['medium'])} medium prompts, requested {N_MEDIUM}")
#         N_medium_final = len(buckets["medium"])
#     else:
#         N_medium_final = N_MEDIUM

#     if len(buckets["long"]) < N_LONG:
#         print(f"WARNING: only {len(buckets['long'])} long prompts, requested {N_LONG}")
#         N_long_final = len(buckets["long"])
#     else:
#         N_long_final = N_LONG

#     # Sample from each bucket
#     random.seed(42)
#     selected = []
#     if buckets["short"]:
#         selected += random.sample(buckets["short"], N_short_final)
#     if buckets["medium"]:
#         selected += random.sample(buckets["medium"], N_medium_final)
#     if buckets["long"]:
#         selected += random.sample(buckets["long"], N_long_final)

#     print(f"Total selected prompts: {len(selected)}")

#     # Sort by bucket for readability
#     selected.sort(key=lambda x: x["bucket"])

#     output = {
#         "config": {
#             "tokenizer_model": TOKENIZER_MODEL_NAME,
#             "short_max_tokens": SHORT_MAX_TOKENS,
#             "medium_max_tokens": MEDIUM_MAX_TOKENS,
#             "n_short": N_short_final,
#             "n_medium": N_medium_final,
#             "n_long": N_long_final,
#             "source_dataset": "lmsys/chatbot_arena_conversations/train",
#             "generated_at": datetime.utcnow().isoformat() + "Z",
#         },
#         "prompts": selected,
#     }

#     with open(OUTPUT_FILE, "w") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)

#     print(f"Saved prompt set to {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()
