# TokenForge GPU-Accelerated LLM Inference Platform
"""
Speculative decoding benchmark.

Compares standard autoregressive decoding against speculative
decoding, measuring speedup, acceptance rate, and tokens/sec.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config
from speculative_decoding.decoder import SpeculativeDecoder

console = Console()


def _standard_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Standard autoregressive generation as baseline."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = output.shape[1] - input_ids.shape[1]

    return {
        "total_time_s": elapsed,
        "tokens_generated": new_tokens,
        "tokens_per_sec": new_tokens / max(elapsed, 1e-9),
    }


def run_speculative_benchmark(
    draft_model: Optional[str] = None,
    target_model: Optional[str] = None,
    prompt: str = "Explain the concept of speculative execution in modern CPUs and how it relates to branch prediction.",
    max_new_tokens: int = 64,
    num_speculative: int = 5,
    num_runs: int = 3,
):
    """Compare standard decoding vs speculative decoding."""
    cfg = get_config()
    draft_model = draft_model or cfg.models.draft_model
    target_model = target_model or cfg.models.target_model

    console.print(f"\n[bold magenta]{'='*60}[/]")
    console.print(f"[bold magenta]Speculative Decoding Benchmark[/]")
    console.print(f"[bold magenta]Draft: {draft_model}[/]")
    console.print(f"[bold magenta]Target: {target_model}[/]")
    console.print(f"[bold magenta]K={num_speculative} speculative tokens[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # Standard baseline
    console.print("[cyan]→ Standard autoregressive decoding...[/]")
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        target_model, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    standard_results = []
    for i in range(num_runs):
        result = _standard_generate(model, tokenizer, prompt, max_new_tokens)
        standard_results.append(result)
        console.print(
            f"  Run {i+1}: {result['tokens_per_sec']:.1f} tok/s "
            f"({result['total_time_s']:.3f}s)"
        )

    del model, tokenizer
    torch.cuda.empty_cache()

    # Speculative decoding
    console.print(f"\n[cyan]→ Speculative decoding (K={num_speculative})...[/]")

    spec_results = []
    for i in range(num_runs):
        decoder = SpeculativeDecoder(
            draft_model_name=draft_model,
            target_model_name=target_model,
            num_speculative_tokens=num_speculative,
        )

        output_text, stats = decoder.generate(
            prompt, max_new_tokens=max_new_tokens
        )

        spec_results.append(stats)
        console.print(
            f"  Run {i+1}: {stats['tokens_per_sec']:.1f} tok/s "
            f"(accept rate: {stats['acceptance_rate']:.1%})"
        )

        decoder.cleanup()

    # Aggregate
    avg_standard = sum(r["tokens_per_sec"] for r in standard_results) / len(standard_results)
    avg_spec = sum(r["tokens_per_sec"] for r in spec_results) / len(spec_results)
    avg_accept = sum(r["acceptance_rate"] for r in spec_results) / len(spec_results)

    speedup = avg_spec / max(avg_standard, 1e-9)

    # Results table
    table = Table(title="\nSpeculative Decoding Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Standard", justify="right", style="green")
    table.add_column("Speculative", justify="right", style="yellow")

    table.add_row("Throughput (tok/s)", f"{avg_standard:.1f}", f"{avg_spec:.1f}")
    table.add_row("Speedup", "1.00x", f"{speedup:.2f}x")
    table.add_row("Acceptance Rate", "—", f"{avg_accept:.1%}")
    table.add_row("Spec. Tokens (K)", "—", str(num_speculative))

    console.print(table)

    return {
        "standard_tok_per_sec": avg_standard,
        "speculative_tok_per_sec": avg_spec,
        "speedup": speedup,
        "acceptance_rate": avg_accept,
    }


if __name__ == "__main__":
    run_speculative_benchmark(num_runs=2, max_new_tokens=32)
