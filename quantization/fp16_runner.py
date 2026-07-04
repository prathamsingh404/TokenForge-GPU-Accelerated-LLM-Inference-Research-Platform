# TokenForge GPU-Accelerated LLM Inference Platform
"""
FP16 (half-precision) inference runner.

Serves as the baseline for quantization comparisons.
FP16 is the standard precision for inference on modern GPUs —
good balance of speed and numerical fidelity.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import get_config
from core.metrics import TimingResult
from benchmark_engine.runner import BenchmarkRunner


def load_fp16_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    # Report memory footprint
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Model loaded in FP16: {param_bytes / 1024**2:.0f} MB parameters")

    return model, tokenizer


def run_fp16_benchmark(
    model_name: Optional[str] = None,
    prompt: str = "Describe the mechanisms behind neural network backpropagation.",
    max_new_tokens: int = 128,
    batch_size: int = 1,
    num_runs: int = 10,
):
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    model, tokenizer = load_fp16_model(model_name)
    device = next(model.parameters()).device

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
        name=f"quant-fp16-bs{batch_size}",
        phase="quantization",
        model_name=model_name,
        timed_runs=num_runs,
    )

    result = runner.run(
        benchmark_fn=benchmark_fn,
        batch_size=batch_size,
        quantization="fp16",
    )

    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    run_fp16_benchmark(num_runs=5)
