# TokenForge GPU-Accelerated LLM Inference Platform
"""
KV cache memory analysis.

Measures actual GPU memory consumption of the KV cache as sequence
length grows. Reveals the quadratic memory scaling that makes
long-context inference challenging.
"""

import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config
from kv_cache.cache_manager import estimate_kv_cache_size

console = Console()


def measure_cache_memory_growth(
    model_name: Optional[str] = None,
    sequence_lengths: Optional[list[int]] = None,
):
    """
    Measure actual VRAM usage at different KV cache sizes by
    running prefill at various input lengths.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model
    sequence_lengths = sequence_lengths or [32, 64, 128, 256, 512]

    console.print(f"\n[bold]Cache Memory Analysis: {model_name}[/]\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    # Baseline memory (model loaded, no cache)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    results = []

    for seq_len in sequence_lengths:
        # Create input of target length
        dummy_ids = torch.randint(
            100, 30000, (1, seq_len), device="cuda"
        )

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            outputs = model(dummy_ids, use_cache=True)
            _ = outputs.past_key_values  # Force cache materialization

        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        cache_mb = peak_mb - baseline_mb

        # Theoretical prediction
        est = estimate_kv_cache_size(model_name, max_seq_len=seq_len)

        results.append({
            "seq_len": seq_len,
            "measured_cache_mb": cache_mb,
            "theoretical_mb": est["total_mb"],
            "peak_total_mb": peak_mb,
        })

        # Clean up cache for next iteration
        del outputs
        torch.cuda.empty_cache()

    # Display
    table = Table(title="KV Cache Memory Scaling")
    table.add_column("Seq Length", justify="right", style="cyan")
    table.add_column("Measured Cache (MB)", justify="right", style="green")
    table.add_column("Theoretical (MB)", justify="right", style="yellow")
    table.add_column("Total VRAM (MB)", justify="right", style="magenta")
    table.add_column("Cache / Token (KB)", justify="right")

    for r in results:
        per_token_kb = (r["measured_cache_mb"] * 1024) / max(r["seq_len"], 1)
        table.add_row(
            str(r["seq_len"]),
            f"{r['measured_cache_mb']:.1f}",
            f"{r['theoretical_mb']:.1f}",
            f"{r['peak_total_mb']:.0f}",
            f"{per_token_kb:.2f}",
        )

    console.print(table)
    console.print(f"\n[dim]Model baseline: {baseline_mb:.0f} MB[/]")

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    measure_cache_memory_growth()
