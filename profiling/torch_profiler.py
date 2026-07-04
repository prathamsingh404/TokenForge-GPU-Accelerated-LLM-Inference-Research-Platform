# TokenForge GPU-Accelerated LLM Inference Platform
"""
PyTorch Profiler wrapper for CUDA kernel analysis.

Alternative to nsys/nvprof that works on Windows. Uses
torch.profiler to trace CUDA activities, kernel launches,
and memory operations. Generates Chrome trace files for
visualization in chrome://tracing.
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config

console = Console()


def profile_inference(
    model_name: Optional[str] = None,
    prompt: str = "The fundamental theorem of calculus connects differentiation and integration.",
    max_new_tokens: int = 16,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Profile a model's inference with full CUDA kernel tracing.
    Generates a Chrome trace file that can be opened in chrome://tracing.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model
    output_dir = output_dir or cfg.profiling_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]CUDA Profiler: {model_name}[/]\n")

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
        model.generate(**inputs, max_new_tokens=4, do_sample=False)

    # Profile
    trace_path = output_dir / f"trace_{model_name.replace('/', '_')}"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(trace_path)),
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Analyze results
    key_averages = prof.key_averages()

    # Build summary
    cuda_time_total = sum(
        getattr(evt, 'cuda_time_total', getattr(evt, 'cpu_time_total', 0)) for evt in key_averages
        if getattr(evt, 'cuda_time_total', getattr(evt, 'cpu_time_total', 0)) > 0
    )

    top_kernels = sorted(
        [e for e in key_averages if e.cuda_time_total > 0],
        key=lambda e: -e.cuda_time_total,
    )[:20]

    # Display
    table = Table(title="Top CUDA Kernels by Time", show_header=True)
    table.add_column("Kernel", style="cyan", max_width=50)
    table.add_column("CUDA Time (ms)", justify="right", style="green")
    table.add_column("% Total", justify="right", style="yellow")
    table.add_column("Calls", justify="right")
    table.add_column("Avg (μs)", justify="right")

    for evt in top_kernels:
        pct = (evt.cuda_time_total / cuda_time_total * 100) if cuda_time_total > 0 else 0
        avg_us = evt.cuda_time_total / max(evt.count, 1)
        table.add_row(
            evt.key[:50],
            f"{evt.cuda_time_total / 1000:.2f}",
            f"{pct:.1f}%",
            str(evt.count),
            f"{avg_us:.1f}",
        )

    console.print(table)
    console.print(f"\n[dim]Trace saved to: {trace_path}[/]")
    console.print("[dim]Open in chrome://tracing or TensorBoard[/]")

    # Categorize by operation type
    categories = _categorize_kernels(top_kernels, cuda_time_total)

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "trace_path": str(trace_path),
        "total_cuda_time_ms": cuda_time_total / 1000,
        "top_kernels": [
            {"name": e.key, "cuda_ms": e.cuda_time_total / 1000, "calls": e.count}
            for e in top_kernels
        ],
        "categories": categories,
    }


def _categorize_kernels(kernels, total_time: float) -> dict:
    """Group kernels into high-level categories."""
    categories = {
        "Attention/GEMM": 0,
        "Softmax": 0,
        "LayerNorm": 0,
        "Elementwise": 0,
        "Memory": 0,
        "Other": 0,
    }

    for evt in kernels:
        name = evt.key.lower()
        time_ms = evt.cuda_time_total / 1000

        if any(k in name for k in ["gemm", "matmul", "mm_", "cublas", "attention", "bmm"]):
            categories["Attention/GEMM"] += time_ms
        elif "softmax" in name:
            categories["Softmax"] += time_ms
        elif "norm" in name or "layer_norm" in name:
            categories["LayerNorm"] += time_ms
        elif any(k in name for k in ["add", "mul", "gelu", "silu", "relu", "elementwise"]):
            categories["Elementwise"] += time_ms
        elif any(k in name for k in ["memcpy", "memset", "copy"]):
            categories["Memory"] += time_ms
        else:
            categories["Other"] += time_ms

    # Print category breakdown
    total_ms = total_time / 1000
    table = Table(title="Kernel Category Breakdown")
    table.add_column("Category", style="cyan")
    table.add_column("Time (ms)", justify="right", style="green")
    table.add_column("% Total", justify="right", style="yellow")

    for cat, ms in sorted(categories.items(), key=lambda x: -x[1]):
        if ms > 0:
            pct = (ms / total_ms * 100) if total_ms > 0 else 0
            table.add_row(cat, f"{ms:.2f}", f"{pct:.1f}%")

    console.print(table)
    return categories


if __name__ == "__main__":
    profile_inference(max_new_tokens=8)
