# TokenForge GPU-Accelerated LLM Inference Platform
"""
KV cache benchmark: cache ON vs cache OFF.

Demonstrates the enormous performance impact of KV caching
by comparing generation with and without cache reuse.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config
from core.metrics import TimingResult
from core.gpu_monitor import GPUMonitor
from benchmark_engine.runner import BenchmarkRunner
from kv_cache.cache_manager import estimate_kv_cache_size

console = Console()


def _generate_no_cache(model, input_ids, max_new_tokens):
    """
    Generate tokens WITHOUT KV cache — recomputes all attention
    for every new token. Extremely slow, but demonstrates why
    caching matters.
    """
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def _generate_with_cache(model, input_ids, max_new_tokens):
    """
    Generate tokens WITH KV cache — standard efficient generation.
    Only processes the new token at each step, reusing cached
    key/value tensors from previous steps.
    """
    generated = input_ids.clone()
    past_key_values = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(generated, use_cache=True)
            else:
                outputs = model(
                    generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def run_kv_cache_benchmark(
    model_name: Optional[str] = None,
    prompt: str = "Explain how KV caching accelerates autoregressive generation in transformers.",
    max_new_tokens: int = 32,
    num_runs: int = 5,
):
    """Compare generation speed with and without KV cache."""
    cfg = get_config()
    model_name = model_name or cfg.models.small_model  # Use small model — no-cache is very slow

    console.print(f"\n[bold magenta]KV Cache Benchmark: {model_name}[/]\n")

    # Show theoretical cache size
    cache_est = estimate_kv_cache_size(model_name, max_seq_len=512)
    console.print(f"[dim]Estimated KV cache for 512 tokens: {cache_est['total_mb']:.1f} MB[/]")
    console.print(f"[dim]Per-token cache cost: {cache_est['per_token_kb']:.2f} KB[/]\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # --- No cache ---
    console.print("[cyan]Running WITHOUT KV cache...[/]")
    no_cache_times = []
    for i in range(num_runs + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _generate_no_cache(model, input_ids, max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i > 0:  # skip warmup
            no_cache_times.append(elapsed)
        console.print(f"  Run {i}: {elapsed:.3f}s")

    # --- With cache ---
    console.print("\n[cyan]Running WITH KV cache...[/]")
    cache_times = []
    for i in range(num_runs + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _generate_with_cache(model, input_ids, max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i > 0:
            cache_times.append(elapsed)
        console.print(f"  Run {i}: {elapsed:.3f}s")

    # Results
    avg_no_cache = sum(no_cache_times) / len(no_cache_times)
    avg_cache = sum(cache_times) / len(cache_times)
    speedup = avg_no_cache / max(avg_cache, 1e-9)

    table = Table(title="\nKV Cache Impact")
    table.add_column("Mode", style="cyan")
    table.add_column("Avg Time (s)", justify="right", style="green")
    table.add_column("Tokens/sec", justify="right", style="yellow")
    table.add_column("Speedup", justify="right", style="magenta")

    table.add_row(
        "No Cache",
        f"{avg_no_cache:.3f}",
        f"{max_new_tokens / avg_no_cache:.1f}",
        "1.00x",
    )
    table.add_row(
        "With Cache",
        f"{avg_cache:.3f}",
        f"{max_new_tokens / avg_cache:.1f}",
        f"{speedup:.2f}x",
    )

    console.print(table)

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "no_cache_avg_s": avg_no_cache,
        "cache_avg_s": avg_cache,
        "speedup": speedup,
        "tokens_generated": max_new_tokens,
    }


if __name__ == "__main__":
    run_kv_cache_benchmark(num_runs=3)
