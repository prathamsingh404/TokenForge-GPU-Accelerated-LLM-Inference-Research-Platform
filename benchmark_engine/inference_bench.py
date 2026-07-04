# TokenForge GPU-Accelerated LLM Inference Platform
"""
End-to-end inference benchmarks.

Measures throughput (tokens/sec), time-to-first-token (TTFT),
and end-to-end latency for full generation runs. This is the
primary benchmark that most users care about.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import get_config
from core.metrics import TimingResult
from benchmark_engine.runner import BenchmarkRunner


def load_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Load a model and tokenizer with the given precision."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def make_inference_fn(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    batch_size: int = 1,
):
    """
    Create a benchmark callable that runs one inference iteration.

    Returns a function that produces a TimingResult on each call.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    def run_iteration() -> TimingResult:
        torch.cuda.synchronize()

        # Start timing
        start = time.perf_counter()

        # First-token timing: generate just 1 token
        with torch.no_grad():
            first_out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )

        torch.cuda.synchronize()
        ttft = time.perf_counter() - start

        # Full generation
        gen_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        total_time = time.perf_counter() - gen_start

        # Count generated tokens (subtract input length)
        total_new_tokens = sum(
            (outputs[i] != tokenizer.pad_token_id).sum().item() - input_length
            for i in range(outputs.shape[0])
        )
        total_new_tokens = max(total_new_tokens, 1)

        return TimingResult(
            total_time_s=total_time,
            tokens_generated=total_new_tokens,
            time_to_first_token_s=ttft,
            input_tokens=input_length * batch_size,
        )

    return run_iteration


def run_inference_benchmark(
    model_name: Optional[str] = None,
    prompt: str = "Explain the concept of attention mechanisms in transformer architectures.",
    max_new_tokens: int = 128,
    batch_size: int = 1,
    num_runs: int = 10,
    warmup: int = 3,
    quantization: str = "fp16",
):
    """
    Run a full inference benchmark with GPU monitoring.

    This is the high-level entry point that most experiment scripts call.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    # Determine dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(quantization, torch.float16)

    model, tokenizer = load_model_and_tokenizer(model_name, dtype=dtype)
    inference_fn = make_inference_fn(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    runner = BenchmarkRunner(
        name=f"inference-{quantization}-bs{batch_size}",
        phase="inference",
        model_name=model_name,
        warmup_runs=warmup,
        timed_runs=num_runs,
    )

    result = runner.run(
        benchmark_fn=inference_fn,
        batch_size=batch_size,
        quantization=quantization,
    )

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    run_inference_benchmark(num_runs=5, warmup=2)
