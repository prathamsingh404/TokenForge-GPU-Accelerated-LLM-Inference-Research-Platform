"""
Roofline model analysis for GPU workloads.

The roofline model plots achievable performance (FLOP/s) against
operational intensity (FLOP/byte). It shows whether a workload
is memory-bound or compute-bound, which determines the right
optimization strategy.

Used extensively in HPC and GPU kernel optimization.
"""

import torch
import time
from typing import Optional
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from core.config import get_config

console = Console()


@dataclass
class RooflinePoint:
    """A single measurement on the roofline plot."""
    name: str
    flops: float          # Total floating point operations
    bytes_accessed: float  # Total memory traffic (bytes)
    elapsed_s: float       # Wall-clock time

    @property
    def operational_intensity(self) -> float:
        """FLOP per byte of memory traffic."""
        return self.flops / max(self.bytes_accessed, 1)

    @property
    def achieved_flops(self) -> float:
        """Achieved FLOP/s."""
        return self.flops / max(self.elapsed_s, 1e-12)

    @property
    def achieved_bandwidth(self) -> float:
        """Achieved memory bandwidth (bytes/s)."""
        return self.bytes_accessed / max(self.elapsed_s, 1e-12)

    @property
    def is_memory_bound(self) -> bool:
        """Rough heuristic: OI < 10 is usually memory-bound on modern GPUs."""
        return self.operational_intensity < 10


def estimate_gpu_peak_performance(device_index: int = 0) -> dict:
    """
    Estimate theoretical peak FLOP/s and memory bandwidth.
    Uses synthetic benchmarks since we can't query these directly.
    """
    props = torch.cuda.get_device_properties(device_index)

    # Peak FP16 FLOP/s estimate
    # RTX 5050: ~20 SMs × ~128 FP16 cores/SM × 2 GHz ≈ ~5 TFLOP/s
    # This is a rough estimate; actual varies by architecture
    sm_count = props.multi_processor_count

    # Measure actual bandwidth
    size = 256 * 1024 * 1024  # 256 MB
    a = torch.randn(size // 4, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)

    # Warmup
    b.copy_(a)
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    measured_bw = (size * 2 * 10) / elapsed  # read + write

    # Estimate peak compute with matmul
    M = 2048
    mat_a = torch.randn(M, M, device="cuda", dtype=torch.float16)
    mat_b = torch.randn(M, M, device="cuda", dtype=torch.float16)

    # Warmup
    torch.mm(mat_a, mat_b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        torch.mm(mat_a, mat_b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops_per_matmul = 2 * M * M * M  # 2*M^3 for matrix multiply
    measured_flops = (flops_per_matmul * 20) / elapsed

    return {
        "peak_bandwidth_gb_s": measured_bw / 1e9,
        "peak_flops_tflop_s": measured_flops / 1e12,
        "sm_count": sm_count,
        "gpu_name": props.name,
        "ridge_point": measured_flops / measured_bw,  # OI where compute meets bandwidth
    }


def measure_matmul_roofline(
    sizes: Optional[list[int]] = None,
) -> list[RooflinePoint]:
    """
    Measure matmul at various sizes to trace the roofline curve.
    Small matrices are memory-bound; large ones are compute-bound.
    """
    sizes = sizes or [64, 128, 256, 512, 1024, 2048, 4096]
    points = []

    for M in sizes:
        a = torch.randn(M, M, device="cuda", dtype=torch.float16)
        b = torch.randn(M, M, device="cuda", dtype=torch.float16)

        # Warmup
        torch.mm(a, b)
        torch.cuda.synchronize()

        num_iters = max(5, 100 // max(M // 256, 1))
        start = time.perf_counter()
        for _ in range(num_iters):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters

        flops = 2 * M * M * M
        # Memory: read A + B, write C, each M×M elements × 2 bytes (FP16)
        mem_bytes = 3 * M * M * 2

        points.append(RooflinePoint(
            name=f"MatMul {M}×{M}",
            flops=flops,
            bytes_accessed=mem_bytes,
            elapsed_s=elapsed,
        ))

    return points


def run_roofline_analysis():
    """Full roofline analysis with GPU characterization."""
    console.print("\n[bold]Roofline Model Analysis[/]\n")

    # GPU characteristics
    gpu = estimate_gpu_peak_performance()
    console.print(f"GPU: {gpu['gpu_name']}")
    console.print(f"Peak Bandwidth: {gpu['peak_bandwidth_gb_s']:.1f} GB/s")
    console.print(f"Peak Compute: {gpu['peak_flops_tflop_s']:.2f} TFLOP/s (FP16)")
    console.print(f"Ridge Point (OI): {gpu['ridge_point']:.1f} FLOP/byte\n")

    # Measure points
    points = measure_matmul_roofline()

    table = Table(title="Roofline Data Points")
    table.add_column("Operation", style="cyan")
    table.add_column("OI (FLOP/B)", justify="right", style="yellow")
    table.add_column("GFLOP/s", justify="right", style="green")
    table.add_column("BW (GB/s)", justify="right")
    table.add_column("Bound", style="magenta")

    for p in points:
        bound = "Memory" if p.is_memory_bound else "Compute"
        table.add_row(
            p.name,
            f"{p.operational_intensity:.1f}",
            f"{p.achieved_flops / 1e9:.1f}",
            f"{p.achieved_bandwidth / 1e9:.1f}",
            bound,
        )

    console.print(table)

    return {
        "gpu": gpu,
        "points": [
            {
                "name": p.name,
                "oi": p.operational_intensity,
                "gflops": p.achieved_flops / 1e9,
                "bw_gbs": p.achieved_bandwidth / 1e9,
                "bound": "memory" if p.is_memory_bound else "compute",
            }
            for p in points
        ],
    }


if __name__ == "__main__":
    run_roofline_analysis()
