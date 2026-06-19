import time
from collections import deque
from transformers import AutoTokenizer

class TokenMetrics:

    def __init__(self, model_path="./models/tinyllama", window_size=60):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.window_size = window_size
        self.history = deque()

    def record_request(self, prompt: str, output: str):
        now = time.time()

        prompt_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(output))

        self.history.append({
            "timestamp": now,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens
        })

        self._cleanup()

    def _cleanup(self):
        cutoff = time.time() - self.window_size
        while self.history and self.history[0]["timestamp"] < cutoff:
            self.history.popleft()

    def tokens_per_second(self):
        if not self.history:
            return 0.0

        total_tokens = sum(x["total_tokens"] for x in self.history)
        duration = self.history[-1]["timestamp"] - self.history[0]["timestamp"]

        return total_tokens / max(duration, 1)
