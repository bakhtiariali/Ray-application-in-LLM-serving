import math
import os

import psutil

MODEL_PATH = "./models/tinyllama"
DEVICE = os.getenv("LLM_DEVICE", "cpu")
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
DO_SAMPLE = True
PROMPT_TEMPLATE = "User: {message}\nAssistant:"
WINDOW_SIZE_S = 60
CONTROL_INTERVAL_S = 5
SLO_P95_MS = float(os.getenv("LLM_SLO_P95_MS", "30000"))
TARGET_REQUESTS_PER_REPLICA = 2.0
SAFETY_MARGIN = 1.2

STATIC_REPLICAS_CAP = 4

TORCH_NUM_THREADS = max(1, (os.cpu_count() or STATIC_REPLICAS_CAP) // STATIC_REPLICAS_CAP)

_SAFETENSORS_PATH = os.path.join(MODEL_PATH, "model.safetensors")


def static_max_replicas():
    env_override = os.getenv("LLM_MAX_REPLICAS")
    if env_override is not None:
        return max(1, int(env_override))

    cpu_count = os.cpu_count() or STATIC_REPLICAS_CAP
    ceiling_cpu = cpu_count // TORCH_NUM_THREADS

    safetensors_file_gb = 0
    if os.path.exists(_SAFETENSORS_PATH):
        safetensors_file_gb = (os.path.getsize(_SAFETENSORS_PATH) / (1024**3))
    per_replica_gb = safetensors_file_gb * 2.0 * 1.15

    if per_replica_gb <= 0:
        ceiling_ram = 1
    else:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        ceiling_ram = math.floor(total_ram_gb * 0.65 / per_replica_gb)

    return max(1, min(min(ceiling_cpu, ceiling_ram), STATIC_REPLICAS_CAP))
