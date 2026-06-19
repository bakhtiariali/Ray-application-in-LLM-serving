import torch
from fastapi import Request
from ray import serve
from transformers import AutoModelForCausalLM, AutoTokenizer


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}
)
class TinyLlamaDeployment:

    def __init__(self):

        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map="cpu"
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        ctx = serve.get_replica_context()
        self.replica_id = ctx.replica_tag

        # replica counters
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0


    async def __call__(self, request: Request):

        body = await request.json()
        user_prompt = body.get("message", "")

        prompt = f"""<|system|>
You are a helpful AI assistant.
<|user|>
{user_prompt}
<|assistant|>
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        input_tokens = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = outputs[0]

        generated_ids = output_ids[input_tokens:]
        output_tokens = len(generated_ids)

        response_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        # update counters
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return {
            "response": response_text,
            "meta": {
                "replica_id": self.replica_id,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "replica_total_requests": self.total_requests,
                "replica_total_input_tokens": self.total_input_tokens,
                "replica_total_output_tokens": self.total_output_tokens,
            }
        }


deployment = TinyLlamaDeployment.bind()
