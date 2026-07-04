# TokenForge GPU-Accelerated LLM Inference Platform
"""
Continuous batching benchmark.

Compares static batching (wait-then-process) against continuous
batching (iteration-level scheduling) on identical workloads.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config
from continuous_batching.engine import ContinuousBatchingEngine
from continuous_batching.request_queue import InferenceRequest
from batching.static_batcher import StaticBatcher, SAMPLE_PROMPTS

console = Console()


def _run_static_baseline(
    model, tokenizer, prompts: list[str],
    batch_size: int, max_new_tokens: int,
) -> dict:
    """Run all prompts using static batching."""
    batcher = StaticBatcher(model, tokenizer, batch_size=batch_size)
    start = time.monotonic()
    outputs, total_time = batcher.process(prompts, max_new_tokens=max_new_tokens)
    end = time.monotonic()

    total_tokens = sum(len(tokenizer.encode(o)) for o in outputs)

    return {
        "total_time": end - start,
        "total_tokens": total_tokens,
        "throughput": total_tokens / max(total_time, 1e-9),
        "num_requests": len(prompts),
    }


def _run_continuous(
    model_name: str, prompts: list[str],
    max_batch_size: int, max_new_tokens: int,
) -> dict:
    """Run all prompts using continuous batching."""
    engine = ContinuousBatchingEngine(
        model_name=model_name,
        max_batch_size=max_batch_size,
    )

    requests = [
        InferenceRequest(prompt=p, max_new_tokens=max_new_tokens)
        for p in prompts
    ]

    start = time.monotonic()
    engine.submit(requests)
    completed = engine.run_to_completion()
    end = time.monotonic()

    total_tokens = sum(r.generated_tokens for r in requests)
    ttfts = [r.ttft for r in requests if r.ttft is not None]

    engine.cleanup()

    return {
        "total_time": end - start,
        "total_tokens": total_tokens,
        "throughput": total_tokens / max(end - start, 1e-9),
        "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
        "num_requests": len(prompts),
    }


def run_batching_comparison(
    model_name: Optional[str] = None,
    num_requests: int = 8,
    max_batch_size: int = 4,
    max_new_tokens: int = 32,
):
    """
    Compare static vs continuous batching on the same workload.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model

    prompts = SAMPLE_PROMPTS[:num_requests]
    while len(prompts) < num_requests:
        prompts.extend(SAMPLE_PROMPTS[:num_requests - len(prompts)])

    console.print(f"\n[bold magenta]{'='*60}[/]")
    console.print(f"[bold magenta]Static vs Continuous Batching[/]")
    console.print(f"[bold magenta]Model: {model_name}[/]")
    console.print(f"[bold magenta]Requests: {num_requests}, Max batch: {max_batch_size}[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # Static baseline
    console.print("[cyan]→ Static batching...[/]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    static_result = _run_static_baseline(
        model, tokenizer, prompts, max_batch_size, max_new_tokens
    )
    del model, tokenizer
    torch.cuda.empty_cache()

    # Continuous
    console.print("\n[cyan]→ Continuous batching...[/]")
    continuous_result = _run_continuous(
        model_name, prompts, max_batch_size, max_new_tokens
    )

    # Results
    table = Table(title="\nBatching Strategy Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Static", justify="right", style="green")
    table.add_column("Continuous", justify="right", style="yellow")

    table.add_row(
        "Total Time (s)",
        f"{static_result['total_time']:.2f}",
        f"{continuous_result['total_time']:.2f}",
    )
    table.add_row(
        "Total Tokens",
        str(static_result["total_tokens"]),
        str(continuous_result["total_tokens"]),
    )
    table.add_row(
        "Throughput (tok/s)",
        f"{static_result['throughput']:.1f}",
        f"{continuous_result['throughput']:.1f}",
    )
    if continuous_result.get("avg_ttft"):
        table.add_row(
            "Avg TTFT (s)",
            "—",
            f"{continuous_result['avg_ttft']:.3f}",
        )

    console.print(table)

    return {
        "static": static_result,
        "continuous": continuous_result,
    }


if __name__ == "__main__":
    run_batching_comparison(num_requests=6, max_new_tokens=24)
