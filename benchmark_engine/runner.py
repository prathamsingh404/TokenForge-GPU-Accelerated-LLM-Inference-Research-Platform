"""
Benchmark orchestrator.

Manages the lifecycle of a benchmark experiment: warmup, timed runs,
GPU monitoring, metric aggregation, and persistence to the database.
"""

import gc
import time
from typing import Callable, Optional

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import get_config
from core.database import ExperimentDB
from core.gpu_monitor import GPUMonitor
from core.metrics import (
    TimingResult,
    BenchmarkResult,
    aggregate_timings,
    save_result_json,
)

console = Console()


class BenchmarkRunner:
    """
    Orchestrates benchmark runs with proper warmup, GPU monitoring,
    and result persistence.

    Usage:
        runner = BenchmarkRunner(
            name="fp16-tinyllama-bs4",
            phase="quantization",
            model_name="TinyLlama 1.1B",
        )
        result = runner.run(benchmark_fn, batch_size=4, quantization="fp16")
    """

    def __init__(
        self,
        name: str,
        phase: str,
        model_name: str,
        warmup_runs: Optional[int] = None,
        timed_runs: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
    ):
        cfg = get_config()
        self.name = name
        self.phase = phase
        self.model_name = model_name
        self.warmup_runs = warmup_runs or cfg.benchmarks.warmup_runs
        self.timed_runs = timed_runs or cfg.benchmarks.timed_runs
        self.cooldown_seconds = cooldown_seconds or cfg.benchmarks.cooldown_seconds
        self.config = cfg

    def _clear_gpu_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run(
        self,
        benchmark_fn: Callable[[], TimingResult],
        batch_size: int = 1,
        quantization: str = "none",
        sequence_length: int = 0,
        extra_params: Optional[dict] = None,
        save_to_db: bool = True,
        save_json: bool = True,
    ) -> BenchmarkResult:
        """
        Execute a full benchmark cycle.

        Args:
            benchmark_fn: Callable that performs one inference iteration and
                          returns a TimingResult.
            batch_size: Batch size used in this run.
            quantization: Precision label (fp16, int8, int4, none).
            sequence_length: Input sequence length.
            extra_params: Any additional parameters to record.
            save_to_db: Whether to persist results to SQLite.
            save_json: Whether to save JSON to experiments directory.

        Returns:
            BenchmarkResult with aggregated statistics.
        """
        console.print(f"\n[bold cyan]{'='*60}[/]")
        console.print(f"[bold]Benchmark:[/] {self.name}")
        console.print(f"[bold]Phase:[/] {self.phase}")
        console.print(f"[bold]Model:[/] {self.model_name}")
        console.print(f"[bold]Batch Size:[/] {batch_size}")
        console.print(f"[bold]Quantization:[/] {quantization}")
        console.print(f"[bold cyan]{'='*60}[/]\n")

        # Warmup
        self._clear_gpu_cache()
        console.print(f"[dim]Warming up ({self.warmup_runs} runs)...[/]")
        for _ in range(self.warmup_runs):
            benchmark_fn()

        # Let GPU settle
        time.sleep(self.cooldown_seconds)
        self._clear_gpu_cache()

        # Timed runs with GPU monitoring
        timings: list[TimingResult] = []
        gpu_monitor = GPUMonitor(
            poll_interval_ms=self.config.benchmarks.gpu_poll_interval_ms
        )

        with gpu_monitor:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Running {self.timed_runs} timed iterations...",
                    total=self.timed_runs,
                )
                for i in range(self.timed_runs):
                    result = benchmark_fn()
                    timings.append(result)
                    progress.update(task, advance=1)

        gpu_summary = gpu_monitor.summarize()

        # Aggregate
        result = aggregate_timings(
            timings=timings,
            experiment_name=self.name,
            model_name=self.model_name,
            phase=self.phase,
            gpu_summary=gpu_summary,
            batch_size=batch_size,
            quantization=quantization,
            sequence_length=sequence_length,
        )
        result.batch_size = batch_size
        result.quantization = quantization
        result.sequence_length = sequence_length

        # Display results
        self._print_results(result)

        # Persist
        if save_to_db:
            self._save_to_db(result, extra_params)
        if save_json:
            path = save_result_json(result, self.config.experiments_dir)
            console.print(f"[dim]Saved to {path}[/]")

        self._clear_gpu_cache()
        return result

    def _print_results(self, result: BenchmarkResult):
        table = Table(title="Benchmark Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Throughput (avg)", f"{result.avg_tokens_per_sec:.1f} tok/s")
        table.add_row("Throughput (peak)", f"{result.peak_tokens_per_sec:.1f} tok/s")
        table.add_row("E2E Latency (mean)", f"{result.end_to_end_latency.mean:.4f} s")
        table.add_row("E2E Latency (p95)", f"{result.end_to_end_latency.p95:.4f} s")
        table.add_row("E2E Latency (p99)", f"{result.end_to_end_latency.p99:.4f} s")

        if result.ttft:
            table.add_row("TTFT (mean)", f"{result.ttft.mean:.4f} s")
            table.add_row("TTFT (p95)", f"{result.ttft.p95:.4f} s")

        if result.gpu_summary:
            gs = result.gpu_summary
            table.add_row("GPU Util (avg)", f"{gs.avg_util:.1f}%")
            table.add_row("VRAM Peak", f"{gs.peak_vram_mb:.0f} MB")
            table.add_row("Power (avg)", f"{gs.avg_power_w:.1f} W")
            table.add_row("Temp (max)", f"{gs.max_temperature:.0f}°C")

        table.add_row("Runs", str(result.num_runs))
        console.print(table)

    def _save_to_db(self, result: BenchmarkResult, extra_params: Optional[dict]):
        with ExperimentDB() as db:
            exp_id = db.create_experiment(
                name=self.name,
                phase=self.phase,
                model_name=self.model_name,
                batch_size=result.batch_size,
                quantization=result.quantization,
                sequence_length=result.sequence_length,
                extra_params=extra_params,
            )

            metrics = [
                ("throughput_avg", result.avg_tokens_per_sec, "tokens/sec"),
                ("throughput_peak", result.peak_tokens_per_sec, "tokens/sec"),
                ("e2e_latency_mean", result.end_to_end_latency.mean, "seconds"),
                ("e2e_latency_p95", result.end_to_end_latency.p95, "seconds"),
                ("e2e_latency_p99", result.end_to_end_latency.p99, "seconds"),
                ("e2e_latency_std", result.end_to_end_latency.std, "seconds"),
            ]

            if result.ttft:
                metrics.extend([
                    ("ttft_mean", result.ttft.mean, "seconds"),
                    ("ttft_p95", result.ttft.p95, "seconds"),
                    ("ttft_p99", result.ttft.p99, "seconds"),
                ])

            if result.gpu_summary:
                gs = result.gpu_summary
                metrics.extend([
                    ("gpu_util_avg", gs.avg_util, "percent"),
                    ("gpu_util_max", gs.max_util, "percent"),
                    ("vram_avg_mb", gs.avg_vram_mb, "MB"),
                    ("vram_peak_mb", gs.peak_vram_mb, "MB"),
                    ("power_avg_w", gs.avg_power_w, "watts"),
                    ("power_max_w", gs.max_power_w, "watts"),
                    ("temp_max_c", gs.max_temperature, "celsius"),
                ])

            db.record_metrics_batch(exp_id, metrics)

            # Save GPU snapshots
            if result.gpu_summary and result.gpu_summary.snapshots:
                for snap in result.gpu_summary.snapshots:
                    db.record_gpu_snapshot(
                        experiment_id=exp_id,
                        gpu_util=snap.gpu_util_percent,
                        vram_used_mb=snap.vram_used_mb,
                        vram_total_mb=snap.vram_total_mb,
                        temperature=snap.temperature_c,
                        power_draw=snap.power_draw_w,
                        clock_mhz=snap.clock_mhz,
                    )

            console.print(f"[dim]Experiment {exp_id} saved to database[/]")
