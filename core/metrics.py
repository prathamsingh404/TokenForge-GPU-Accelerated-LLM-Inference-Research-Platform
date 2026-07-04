# TokenForge GPU-Accelerated LLM Inference Platform
"""
Metric collection and statistical aggregation.

Provides `BenchmarkResult` for storing raw timing data, and functions
for computing percentiles and summary statistics over repeated runs.

Metrics tracked:
- Throughput (tokens/sec)
- TTFT (time to first token)
- Time per output token (TPOT)
- End-to-end latency with P50/P90/P95/P99 percentiles
- Energy efficiency (joules/token, tokens/watt)
- Queue wait time (for workload simulation)
- Request completion distribution
- Full environment manifest for reproducibility
"""

import time
import statistics
import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from core.gpu_monitor import GPUMetricsSummary


@dataclass
class TimingResult:
    """Raw timing from a single benchmark iteration."""
    total_time_s: float
    tokens_generated: int
    time_to_first_token_s: Optional[float] = None
    input_tokens: int = 0


@dataclass
class LatencyStats:
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    std: float
    min: float
    max: float


@dataclass
class BenchmarkResult:
    """Aggregated result from multiple timed iterations."""
    experiment_name: str
    model_name: str
    phase: str

    # throughput
    avg_tokens_per_sec: float
    peak_tokens_per_sec: float

    # latency
    end_to_end_latency: LatencyStats
    ttft: Optional[LatencyStats] = None

    # per-token timing
    time_per_output_token: Optional[LatencyStats] = None

    # GPU
    gpu_summary: Optional[GPUMetricsSummary] = None

    # energy efficiency
    total_energy_joules: float = 0.0
    joules_per_token: float = 0.0
    tokens_per_watt: float = 0.0

    # queue metrics (for workload simulation)
    queue_wait_time: Optional[LatencyStats] = None
    request_completion_distribution: Optional[dict] = None

    # reproducibility
    environment: Optional[dict] = None

    # metadata
    batch_size: int = 1
    quantization: str = "none"
    sequence_length: int = 0
    num_runs: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove snapshot list (too large for JSON)
        if d.get("gpu_summary") and "snapshots" in d["gpu_summary"]:
            d["gpu_summary"]["snapshots"] = f"[{d['gpu_summary']['sample_count']} samples]"
        return d


def compute_latency_stats(values: list[float]) -> LatencyStats:
    """Compute summary statistics from a list of latency measurements."""
    if not values:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = int(p / 100.0 * (n - 1))
        return sorted_vals[min(idx, n - 1)]

    return LatencyStats(
        mean=statistics.mean(values),
        median=statistics.median(values),
        p50=percentile(50),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
        std=statistics.stdev(values) if n > 1 else 0.0,
        min=min(values),
        max=max(values),
    )


def aggregate_timings(
    timings: list[TimingResult],
    experiment_name: str,
    model_name: str,
    phase: str,
    gpu_summary: Optional[GPUMetricsSummary] = None,
    **extra,
) -> BenchmarkResult:
    """Build a BenchmarkResult from raw timing data."""
    e2e_latencies = [t.total_time_s for t in timings]
    ttft_values = [t.time_to_first_token_s for t in timings if t.time_to_first_token_s is not None]

    throughputs = []
    for t in timings:
        if t.total_time_s > 0:
            throughputs.append(t.tokens_generated / t.total_time_s)

    return BenchmarkResult(
        experiment_name=experiment_name,
        model_name=model_name,
        phase=phase,
        avg_tokens_per_sec=statistics.mean(throughputs) if throughputs else 0,
        peak_tokens_per_sec=max(throughputs) if throughputs else 0,
        end_to_end_latency=compute_latency_stats(e2e_latencies),
        ttft=compute_latency_stats(ttft_values) if ttft_values else None,
        gpu_summary=gpu_summary,
        num_runs=len(timings),
        extra=extra,
    )


def save_result_json(result: BenchmarkResult, output_dir: Path) -> Path:
    """Save a benchmark result as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    filename = f"{result.experiment_name}_{ts}.json"
    path = output_dir / filename

    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    return path


def save_results_csv(results: list[BenchmarkResult], output_path: Path):
    """Save multiple benchmark results as a CSV for easy analysis."""
    if not results:
        return

    fieldnames = [
        "experiment_name", "model_name", "phase", "batch_size", "quantization",
        "avg_tokens_per_sec", "peak_tokens_per_sec",
        "e2e_mean", "e2e_p95", "e2e_p99",
        "ttft_mean", "ttft_p95",
        "gpu_avg_util", "gpu_peak_vram_mb", "gpu_avg_power_w",
        "num_runs",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "experiment_name": r.experiment_name,
                "model_name": r.model_name,
                "phase": r.phase,
                "batch_size": r.batch_size,
                "quantization": r.quantization,
                "avg_tokens_per_sec": f"{r.avg_tokens_per_sec:.2f}",
                "peak_tokens_per_sec": f"{r.peak_tokens_per_sec:.2f}",
                "e2e_mean": f"{r.end_to_end_latency.mean:.4f}",
                "e2e_p95": f"{r.end_to_end_latency.p95:.4f}",
                "e2e_p99": f"{r.end_to_end_latency.p99:.4f}",
                "ttft_mean": f"{r.ttft.mean:.4f}" if r.ttft else "",
                "ttft_p95": f"{r.ttft.p95:.4f}" if r.ttft else "",
                "gpu_avg_util": f"{r.gpu_summary.avg_util:.1f}" if r.gpu_summary else "",
                "gpu_peak_vram_mb": f"{r.gpu_summary.peak_vram_mb:.0f}" if r.gpu_summary else "",
                "gpu_avg_power_w": f"{r.gpu_summary.avg_power_w:.1f}" if r.gpu_summary else "",
                "num_runs": r.num_runs,
            }
            writer.writerow(row)
