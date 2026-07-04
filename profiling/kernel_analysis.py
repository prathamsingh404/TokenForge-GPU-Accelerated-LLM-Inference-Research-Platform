# TokenForge GPU-Accelerated LLM Inference Platform
"""
CUDA kernel statistics analyzer.

Extracts and analyzes kernel-level performance data from
PyTorch profiler traces. Identifies the most time-consuming
CUDA kernels and their characteristics.
"""

import torch
from torch.profiler import profile, ProfilerActivity
from typing import Optional
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config

console = Console()


def analyze_cuda_kernels(
    model_name: Optional[str] = None,
    prompt: str = "Describe the GPU execution model.",
    max_new_tokens: int = 8,
) -> dict:
    """
    Extract detailed CUDA kernel statistics.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model

    console.print(f"\n[bold]CUDA Kernel Analysis: {model_name}[/]\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.no_grad():
        model(**inputs)

    # Profile with kernel-level detail
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Extract kernel stats
    events = prof.key_averages()

    # Separate CUDA kernels from CPU ops
    cuda_kernels = [
        e for e in events
        if e.cuda_time_total > 0
    ]

    # Sort by time
    cuda_kernels.sort(key=lambda e: -e.cuda_time_total)
    total_cuda = sum(e.cuda_time_total for e in cuda_kernels)

    # Memory analysis
    total_cuda_mem = sum(
        e.cuda_memory_usage for e in events if e.cuda_memory_usage > 0
    )

    # Display top kernels
    table = Table(title="CUDA Kernel Statistics")
    table.add_column("Kernel", style="cyan", max_width=40)
    table.add_column("Time (ms)", justify="right", style="green")
    table.add_column("%", justify="right", style="yellow")
    table.add_column("Calls", justify="right")
    table.add_column("Avg (μs)", justify="right")
    table.add_column("Input Shape", style="dim", max_width=30)

    for evt in cuda_kernels[:15]:
        pct = (evt.cuda_time_total / total_cuda * 100) if total_cuda > 0 else 0
        avg_us = evt.cuda_time_total / max(evt.count, 1)
        shapes = str(evt.input_shapes)[:30] if evt.input_shapes else "—"

        table.add_row(
            evt.key[:40],
            f"{evt.cuda_time_total / 1000:.2f}",
            f"{pct:.1f}",
            str(evt.count),
            f"{avg_us:.0f}",
            shapes,
        )

    console.print(table)

    # Summary stats
    console.print(f"\n[dim]Total CUDA time: {total_cuda / 1000:.2f} ms[/]")
    console.print(f"[dim]Unique kernels: {len(cuda_kernels)}[/]")
    console.print(f"[dim]Total kernel calls: {sum(e.count for e in cuda_kernels)}[/]")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "total_cuda_ms": total_cuda / 1000,
        "unique_kernels": len(cuda_kernels),
        "total_calls": sum(e.count for e in cuda_kernels),
        "top_kernels": [
            {
                "name": e.key,
                "cuda_ms": e.cuda_time_total / 1000,
                "pct": (e.cuda_time_total / total_cuda * 100) if total_cuda > 0 else 0,
                "calls": e.count,
            }
            for e in cuda_kernels[:15]
        ],
    }


if __name__ == "__main__":
    analyze_cuda_kernels()
