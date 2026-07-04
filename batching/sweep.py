# TokenForge GPU-Accelerated LLM Inference Platform
"""
Batch size sweep experiment.

Systematically varies batch size from 1 to the GPU's saturation point,
measuring throughput and latency at each level. This reveals the
batch size vs throughput curve — critical for capacity planning.
"""

import gc
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config
from core.metrics import TimingResult, BenchmarkResult
from core.database import ExperimentDB
from benchmark_engine.runner import BenchmarkRunner
from batching.static_batcher import StaticBatcher, SAMPLE_PROMPTS

console = Console()


def run_batch_sweep(
    model_name: Optional[str] = None,
    batch_sizes: Optional[list[int]] = None,
    max_new_tokens: int = 64,
    num_runs: int = 5,
    quantization: str = "fp16",
) -> list[BenchmarkResult]:
    """
    Sweep through batch sizes and measure throughput at each.
    Automatically stops on OOM.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model
    batch_sizes = batch_sizes or cfg.benchmarks.batch_sizes

    console.print(f"\n[bold magenta]{'='*60}[/]")
    console.print(f"[bold magenta]Batch Size Sweep: {model_name}[/]")
    console.print(f"[bold magenta]Batch sizes: {batch_sizes}[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # Load model once
    dtype = torch.float16 if quantization == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.eval()

    results = []

    for bs in batch_sizes:
        console.print(f"\n[cyan]→ Batch size {bs}...[/]")

        try:
            batcher = StaticBatcher(model, tokenizer, batch_size=bs)

            def benchmark_fn() -> TimingResult:
                return batcher.benchmark_single_batch(
                    prompts=SAMPLE_PROMPTS[:bs],
                    max_new_tokens=max_new_tokens,
                )

            runner = BenchmarkRunner(
                name=f"batch-sweep-bs{bs}",
                phase="batching",
                model_name=model_name,
                warmup_runs=2,
                timed_runs=num_runs,
            )

            result = runner.run(
                benchmark_fn=benchmark_fn,
                batch_size=bs,
                quantization=quantization,
            )
            results.append(result)

        except torch.cuda.OutOfMemoryError:
            console.print(f"[red]OOM at batch size {bs} — stopping sweep[/]")
            gc.collect()
            torch.cuda.empty_cache()
            break

        except Exception as e:
            console.print(f"[red]Error at batch size {bs}: {e}[/]")
            gc.collect()
            torch.cuda.empty_cache()
            break

    # Summary table
    if results:
        _print_sweep_summary(results)

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


def _print_sweep_summary(results: list[BenchmarkResult]):
    table = Table(title="\nBatch Size Sweep Summary", show_header=True)
    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("Latency mean (ms)", justify="right", style="yellow")
    table.add_column("Latency p95 (ms)", justify="right")
    table.add_column("VRAM Peak (MB)", justify="right", style="magenta")
    table.add_column("GPU Util %", justify="right")

    for r in results:
        vram = f"{r.gpu_summary.peak_vram_mb:.0f}" if r.gpu_summary else "—"
        util = f"{r.gpu_summary.avg_util:.0f}" if r.gpu_summary else "—"
        table.add_row(
            str(r.batch_size),
            f"{r.avg_tokens_per_sec:.1f}",
            f"{r.end_to_end_latency.mean * 1000:.0f}",
            f"{r.end_to_end_latency.p95 * 1000:.0f}",
            vram,
            util,
        )

    console.print(table)

    # Find sweet spot
    if len(results) >= 2:
        best = max(results, key=lambda r: r.avg_tokens_per_sec)
        console.print(
            f"\n[bold green]Optimal batch size: {best.batch_size} "
            f"({best.avg_tokens_per_sec:.1f} tok/s)[/]"
        )


if __name__ == "__main__":
    run_batch_sweep(num_runs=3)
