import ray
from ray import serve
from fastapi import Request

ray.init()
serve.start()


@serve.deployment
class TinyLlamaDeployment:

    def __init__(self):

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch

        self.model_path = "./models/tinyllama"

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded on {self.device}")

    async def __call__(self, request: Request):

        data = await request.json()
        message = data.get("message", "")

        inputs = self.tokenizer(
            f"User: {message}\nAssistant:",
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return {"response": text.strip()}


app = TinyLlamaDeployment.bind()

serve.run(app, route_prefix="/chat")

input("Press Enter to stop...")