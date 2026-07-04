"""
Tensor Parallelism simulation.

Simulates tensor parallelism by splitting weight matrices along
their appropriate dimensions and computing each partition sequentially
on a single GPU. Logs communication cost (all-reduce) that would
occur on multi-GPU setups.

This enables studying TP scaling behavior and communication overhead
without requiring multi-GPU hardware.

Key concepts:
- Column parallel: Split the weight matrix along columns (output dim).
  Each GPU computes a different slice of the output. Requires all-gather.
- Row parallel: Split along rows (input dim). Each GPU gets full input
  but partial weights. Requires all-reduce after computation.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TPSimulationResult:
    """Results from a tensor parallelism simulation run."""
    tp_degree: int
    num_layers: int
    hidden_size: int

    # Compute time per partition
    partition_times_ms: list[float] = field(default_factory=list)

    # Simulated communication cost
    all_reduce_bytes: int = 0
    estimated_all_reduce_ms: float = 0.0

    # Total estimated time
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Scaling analysis
    ideal_speedup: float = 0.0
    actual_speedup: float = 0.0
    parallel_efficiency: float = 0.0

    # Bandwidth
    inter_gpu_bandwidth_gbps: float = 600.0  # NVLink default


class TensorParallelSimulator:
    """
    Simulate tensor parallelism on a single GPU.

    Splits weight matrices across `tp_degree` virtual GPUs and
    measures the compute time for each partition. Estimates
    communication overhead (all-reduce / all-gather) based on
    configured inter-GPU bandwidth.

    Usage:
        sim = TensorParallelSimulator(model, tp_degree=4)
        result = sim.run()
        print(f"Parallel efficiency: {result.parallel_efficiency:.1%}")
    """

    def __init__(
        self,
        model: nn.Module,
        tp_degree: int = 2,
        inter_gpu_bandwidth_gbps: float = 600.0,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.model = model
        self.tp_degree = tp_degree
        self.bandwidth_gbps = inter_gpu_bandwidth_gbps
        self.dtype = dtype
        self.device = device

    def run(
        self,
        input_ids: Optional[torch.Tensor] = None,
        seq_len: int = 512,
        batch_size: int = 1,
        warmup_iters: int = 3,
        timed_iters: int = 10,
    ) -> TPSimulationResult:
        """
        Run the tensor parallelism simulation.

        1. Run baseline (no split) to measure single-GPU time.
        2. Split each linear layer by tp_degree.
        3. Run each partition sequentially and time it.
        4. Estimate all-reduce communication cost.
        5. Compute parallel efficiency.
        """
        from rich.console import Console
        from rich.table import Table
        console = Console()

        console.print(f"\n[bold cyan]Tensor Parallelism Simulation[/] — TP={self.tp_degree}\n")

        if input_ids is None:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

        result = TPSimulationResult(
            tp_degree=self.tp_degree,
            num_layers=0,
            hidden_size=0,
            inter_gpu_bandwidth_gbps=self.bandwidth_gbps,
        )

        # Identify linear layers
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

        result.num_layers = len(linear_layers)
        if linear_layers:
            result.hidden_size = linear_layers[0][1].in_features

        console.print(f"[dim]Found {len(linear_layers)} linear layers to parallelize[/]")

        # Measure baseline (single GPU, no split)
        baseline_ms = self._benchmark_forward(input_ids, warmup_iters, timed_iters)
        console.print(f"[dim]Baseline forward: {baseline_ms:.2f} ms[/]")

        # Simulate partitioned execution
        total_compute = 0.0
        total_comm_bytes = 0

        table = Table(title=f"TP-{self.tp_degree} Layer Analysis", show_lines=False)
        table.add_column("Layer", style="cyan", max_width=40)
        table.add_column("Shape", style="dim")
        table.add_column("Partition Time (ms)", justify="right", style="green")
        table.add_column("All-Reduce (KB)", justify="right", style="yellow")

        for name, module in linear_layers:
            w = module.weight
            out_features, in_features = w.shape

            # Column parallel: split output dim
            partition_size = out_features // self.tp_degree
            if partition_size == 0:
                continue

            # Time one partition
            x = torch.randn(batch_size, seq_len, in_features,
                           dtype=self.dtype, device=self.device)
            w_partition = w[:partition_size, :].to(self.dtype)

            # Warmup
            for _ in range(warmup_iters):
                torch.mm(x.view(-1, in_features), w_partition.t())
            torch.cuda.synchronize()

            # Time
            times = []
            for _ in range(timed_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.mm(x.view(-1, in_features), w_partition.t())
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

            avg_ms = sum(times) / len(times)
            result.partition_times_ms.append(avg_ms)
            total_compute += avg_ms

            # All-reduce cost: each GPU sends partition_size outputs
            # All-reduce transfers 2*(P-1)/P * data_size bytes
            data_bytes = batch_size * seq_len * partition_size * 2  # FP16
            all_reduce_bytes = int(2 * (self.tp_degree - 1) / self.tp_degree * data_bytes)
            total_comm_bytes += all_reduce_bytes

            table.add_row(
                name[:40],
                f"{in_features}×{out_features}",
                f"{avg_ms:.3f}",
                f"{all_reduce_bytes / 1024:.1f}",
            )

            del x, w_partition

        console.print(table)

        # Estimate communication time
        # NVLink bandwidth in bytes/ms = gbps * 1e9 / 8 / 1e3
        bandwidth_bytes_ms = self.bandwidth_gbps * 1e9 / 8 / 1000
        comm_time_ms = total_comm_bytes / bandwidth_bytes_ms

        result.compute_time_ms = total_compute
        result.all_reduce_bytes = total_comm_bytes
        result.estimated_all_reduce_ms = comm_time_ms
        result.communication_time_ms = comm_time_ms
        result.total_time_ms = total_compute + comm_time_ms
        result.ideal_speedup = self.tp_degree
        result.actual_speedup = baseline_ms / result.total_time_ms if result.total_time_ms > 0 else 0
        result.parallel_efficiency = result.actual_speedup / self.tp_degree if self.tp_degree > 0 else 0

        # Summary
        console.print(f"\n[bold]Results:[/]")
        console.print(f"  Compute: {total_compute:.2f} ms")
        console.print(f"  Communication: {comm_time_ms:.2f} ms ({total_comm_bytes / (1024**2):.1f} MB all-reduce)")
        console.print(f"  Total: {result.total_time_ms:.2f} ms")
        console.print(f"  Speedup: {result.actual_speedup:.2f}x (ideal: {self.tp_degree}x)")
        console.print(f"  Parallel Efficiency: {result.parallel_efficiency:.1%}")
        console.print(f"  Compute/Comm Ratio: {total_compute / max(comm_time_ms, 0.001):.1f}\n")

        return result

    def _benchmark_forward(
        self, input_ids: torch.Tensor,
        warmup: int, iters: int,
    ) -> float:
        """Measure baseline forward pass time."""
        with torch.no_grad():
            for _ in range(warmup):
                self.model(input_ids)
            torch.cuda.synchronize()

            times = []
            for _ in range(iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                self.model(input_ids)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        return sum(times) / len(times)
