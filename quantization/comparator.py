"""
Cross-precision comparator.

Runs FP16, INT8, and INT4 benchmarks on the same model and prompt,
then produces a side-by-side comparison of throughput, latency,
and memory usage.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table

from core.config import get_config
from core.metrics import BenchmarkResult, save_results_csv
from quantization.fp16_runner import run_fp16_benchmark
from quantization.int8_runner import run_int8_benchmark
from quantization.int4_runner import run_int4_benchmark

console = Console()


def run_quantization_comparison(
    model_name: Optional[str] = None,
    prompt: str = "Describe the mechanisms behind neural network backpropagation.",
    max_new_tokens: int = 128,
    batch_size: int = 1,
    num_runs: int = 8,
) -> list[BenchmarkResult]:
    """Run FP16 → INT8 → INT4 sequentially and compare."""

    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    console.print(f"\n[bold magenta]{'='*60}[/]")
    console.print(f"[bold magenta]Quantization Comparison: {model_name}[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    results = []

    # FP16 baseline
    console.print("[bold cyan]→ Running FP16 baseline...[/]")
    r_fp16 = run_fp16_benchmark(
        model_name=model_name, prompt=prompt,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
        num_runs=num_runs,
    )
    results.append(r_fp16)

    # INT8
    console.print("\n[bold cyan]→ Running INT8...[/]")
    r_int8 = run_int8_benchmark(
        model_name=model_name, prompt=prompt,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
        num_runs=num_runs,
    )
    results.append(r_int8)

    # INT4
    console.print("\n[bold cyan]→ Running INT4...[/]")
    r_int4 = run_int4_benchmark(
        model_name=model_name, prompt=prompt,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
        num_runs=num_runs,
    )
    results.append(r_int4)

    # Comparison table
    _print_comparison(results)

    # Save CSV
    csv_path = cfg.experiments_dir / "quantization_comparison.csv"
    save_results_csv(results, csv_path)
    console.print(f"\n[dim]Comparison saved to {csv_path}[/]")

    return results


def _print_comparison(results: list[BenchmarkResult]):
    table = Table(
        title="\nQuantization Comparison",
        show_header=True,
        title_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")

    for r in results:
        table.add_column(r.quantization.upper(), justify="right", style="green")

    # Throughput
    table.add_row(
        "Throughput (tok/s)",
        *[f"{r.avg_tokens_per_sec:.1f}" for r in results],
    )

    # Speedup relative to FP16
    if results[0].avg_tokens_per_sec > 0:
        table.add_row(
            "Speedup vs FP16",
            *[f"{r.avg_tokens_per_sec / results[0].avg_tokens_per_sec:.2f}x" for r in results],
        )

    # Latency
    table.add_row(
        "E2E Latency (ms)",
        *[f"{r.end_to_end_latency.mean * 1000:.1f}" for r in results],
    )

    table.add_row(
        "Latency p95 (ms)",
        *[f"{r.end_to_end_latency.p95 * 1000:.1f}" for r in results],
    )

    # GPU
    for r in results:
        if r.gpu_summary is None:
            return  # Skip GPU rows if monitoring wasn't available

    table.add_row(
        "VRAM Peak (MB)",
        *[f"{r.gpu_summary.peak_vram_mb:.0f}" for r in results],
    )

    # Memory savings
    if results[0].gpu_summary.peak_vram_mb > 0:
        table.add_row(
            "Memory Savings",
            *[
                f"{(1 - r.gpu_summary.peak_vram_mb / results[0].gpu_summary.peak_vram_mb) * 100:.0f}%"
                for r in results
            ],
        )

    table.add_row(
        "GPU Util (avg %)",
        *[f"{r.gpu_summary.avg_util:.0f}" for r in results],
    )

    table.add_row(
        "Power (avg W)",
        *[f"{r.gpu_summary.avg_power_w:.1f}" for r in results],
    )

    console.print(table)


if __name__ == "__main__":
    run_quantization_comparison(num_runs=5)
