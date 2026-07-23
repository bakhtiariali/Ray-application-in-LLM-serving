import os
import time

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import Request

import config


class BaseLLMDeployment:
    """
    Shared core deployment for both baseline and token-aware modes.
    D1/D2: CPU/fp32, shared generation config, proper error handling,
    token counts from tensor shapes (no re-tokenization).
    """

    def __init__(self, controller_handle=None):
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_PATH,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.set_num_threads(config.TORCH_NUM_THREADS)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            local_files_only=True,
        )

        self.model.to(config.DEVICE)
        self.model.eval()

        self.controller_handle = controller_handle
        self._proc = psutil.Process()

    async def __call__(self, request: Request):
        payload = await request.json()
        message = payload.get("message", "")

        if not message:
            return {"response": "", "metrics": {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}, "error": "Empty message received."}

        if self.controller_handle is not None:
            try:
                self.controller_handle.record_start.remote()
            except Exception:
                pass

        start = time.time()
        output_tokens = 0
        input_len = 0
        response = ""
        error = None

        try:
            prompt = config.PROMPT_TEMPLATE.format(message=message)
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    padding=True, truncation=True)

            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    do_sample=config.DO_SAMPLE,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            output_tokens = outputs.shape[1] - input_len
            response = self.tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True,
            ).strip()

        except Exception as e:
            error = repr(e)

        finally:
            latency_ms = (time.time() - start) * 1000.0

            if self.controller_handle is not None:
                try:
                    cpu_count = os.cpu_count() or 1
                    self.controller_handle.record.remote(
                        input_len, output_tokens, latency_ms,
                        cpu_util=self._proc.cpu_percent() / 100.0 / cpu_count,
                        rss_bytes=self._proc.memory_info().rss,
                    )
                except Exception:
                    pass

        return {
            "response": response,
            "metrics": {
                "input_tokens": input_len,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
            },
            **({"error": error} if error else {}),
        }
