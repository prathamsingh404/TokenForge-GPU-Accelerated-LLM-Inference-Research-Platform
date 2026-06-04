"""
INT4 quantized inference via bitsandbytes NF4.

4-bit NormalFloat quantization — the precision level used by QLoRA.
Aggressive compression that enables running 7B models on consumer GPUs.
Expect ~3.5x memory reduction compared to FP16 with surprisingly
small quality degradation on most benchmarks.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from core.config import get_config
from core.metrics import TimingResult
from benchmark_engine.runner import BenchmarkRunner


def load_int4_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # nested quantization for extra savings
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="cuda",
    )
    model.eval()

    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"  Model loaded in INT4 (NF4): {mem_allocated:.0f} MB GPU memory allocated")

    return model, tokenizer


def run_int4_benchmark(
    model_name: Optional[str] = None,
    prompt: str = "Describe the mechanisms behind neural network backpropagation.",
    max_new_tokens: int = 128,
    batch_size: int = 1,
    num_runs: int = 10,
):
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    model, tokenizer = load_int4_model(model_name)
    device = next(iter(model.parameters())).device

    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    def benchmark_fn() -> TimingResult:
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = out.shape[1] - input_len
        total_tokens = new_tokens * batch_size

        return TimingResult(
            total_time_s=elapsed,
            tokens_generated=total_tokens,
            input_tokens=input_len * batch_size,
        )

    runner = BenchmarkRunner(
        name=f"quant-int4-bs{batch_size}",
        phase="quantization",
        model_name=model_name,
        timed_runs=num_runs,
    )

    result = runner.run(
        benchmark_fn=benchmark_fn,
        batch_size=batch_size,
        quantization="int4",
    )

    del model, tokenizer
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run_int4_benchmark(num_runs=5)
