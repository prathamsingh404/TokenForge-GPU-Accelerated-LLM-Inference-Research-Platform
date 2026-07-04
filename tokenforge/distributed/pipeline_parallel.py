"""
Pipeline Parallelism simulation.

Simulates pipeline parallelism by partitioning transformer layers
into stages and processing micro-batches with timing analysis.
Measures pipeline bubble overhead, stage utilization, and
throughput scaling.

Key concepts:
- Stage: A contiguous group of transformer layers assigned to one GPU.
- Micro-batch: Input batch split into smaller pieces for pipeline fill.
- Pipeline bubble: Idle time at the start/end of pipeline execution.
- Fill/drain: Phases where the pipeline is warming up / cooling down.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StageProfile:
    """Performance profile for a single pipeline stage."""
    stage_id: int
    layer_range: tuple[int, int]
    num_layers: int
    avg_forward_ms: float = 0.0
    avg_backward_ms: float = 0.0
    memory_mb: float = 0.0


@dataclass
class PPSimulationResult:
    """Results from a pipeline parallelism simulation."""
    pp_degree: int
    num_microbatches: int
    num_layers: int

    # Stage-level timing
    stage_profiles: list[StageProfile] = field(default_factory=list)

    # Pipeline metrics
    total_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    bubble_time_ms: float = 0.0
    bubble_ratio: float = 0.0

    # Scaling
    baseline_time_ms: float = 0.0
    ideal_speedup: float = 0.0
    actual_speedup: float = 0.0
    pipeline_efficiency: float = 0.0

    # Throughput
    throughput_samples_per_sec: float = 0.0


class PipelineParallelSimulator:
    """
    Simulate pipeline parallelism on a single GPU.

    Partitions the model's layers into stages and simulates the
    pipeline schedule (GPipe-style) with configurable micro-batches.

    The simulation measures:
    - Per-stage forward pass time
    - Pipeline bubble overhead
    - Stage load balancing
    - Throughput scaling vs baseline

    Usage:
        sim = PipelineParallelSimulator(model, pp_degree=4, num_microbatches=8)
        result = sim.run()
    """

    def __init__(
        self,
        model: nn.Module,
        pp_degree: int = 4,
        num_microbatches: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.pp_degree = pp_degree
        self.num_microbatches = num_microbatches
        self.device = device
        self.dtype = dtype

    def _get_transformer_layers(self) -> list[nn.Module]:
        """Extract transformer layers from the model."""
        layers = []
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__.lower()
            if ("decoderlayer" in class_name or "block" in class_name
                    or "transformerlayer" in class_name):
                layers.append(module)

        # Fallback: use all children if no transformer layers found
        if not layers:
            children = list(self.model.children())
            for child in children:
                if hasattr(child, '__len__') and len(list(child.children())) > 1:
                    layers = list(child.children())
                    break

        return layers

    def run(
        self,
        seq_len: int = 512,
        batch_size: int = 8,
        warmup_iters: int = 2,
        timed_iters: int = 5,
    ) -> PPSimulationResult:
        """
        Run the pipeline parallelism simulation.

        1. Partition layers into stages
        2. Profile each stage's forward time
        3. Simulate GPipe schedule
        4. Compute bubble overhead and efficiency
        """
        from rich.console import Console
        from rich.table import Table
        console = Console()

        console.print(f"\n[bold cyan]Pipeline Parallelism Simulation[/] — PP={self.pp_degree}\n")

        layers = self._get_transformer_layers()
        num_layers = len(layers)

        console.print(f"[dim]Found {num_layers} transformer layers[/]")

        result = PPSimulationResult(
            pp_degree=self.pp_degree,
            num_microbatches=self.num_microbatches,
            num_layers=num_layers,
        )

        if num_layers == 0:
            console.print("[yellow]No transformer layers found for pipeline simulation.[/]")
            return result

        # Partition layers into stages (balanced)
        layers_per_stage = num_layers // self.pp_degree
        extra = num_layers % self.pp_degree

        stage_profiles = []
        start_layer = 0

        for stage_id in range(self.pp_degree):
            n = layers_per_stage + (1 if stage_id < extra else 0)
            end_layer = start_layer + n

            # Profile this stage
            stage_layers = layers[start_layer:end_layer]
            avg_time = self._profile_stage(
                stage_layers, seq_len, batch_size // self.num_microbatches,
                warmup_iters, timed_iters,
            )

            profile = StageProfile(
                stage_id=stage_id,
                layer_range=(start_layer, end_layer),
                num_layers=n,
                avg_forward_ms=avg_time,
            )
            stage_profiles.append(profile)
            start_layer = end_layer

        result.stage_profiles = stage_profiles

        # Display stage profiles
        table = Table(title="Pipeline Stage Profiles")
        table.add_column("Stage", justify="center", style="cyan")
        table.add_column("Layers", justify="center")
        table.add_column("Forward (ms)", justify="right", style="green")
        table.add_column("Relative Load", justify="right", style="yellow")

        max_time = max(p.avg_forward_ms for p in stage_profiles)
        for p in stage_profiles:
            bar_len = int(20 * p.avg_forward_ms / max_time) if max_time > 0 else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            table.add_row(
                str(p.stage_id),
                f"{p.layer_range[0]}-{p.layer_range[1]}",
                f"{p.avg_forward_ms:.2f}",
                bar,
            )
        console.print(table)

        # GPipe schedule simulation
        # Pipeline timeline: each cell is (stage, microbatch)
        # Total time = (P - 1 + M) * max_stage_time
        # where P = pp_degree, M = num_microbatches
        max_stage_ms = max(p.avg_forward_ms for p in stage_profiles)
        total_compute = sum(p.avg_forward_ms for p in stage_profiles) * self.num_microbatches
        pipeline_time = (self.pp_degree - 1 + self.num_microbatches) * max_stage_ms

        bubble_time = pipeline_time - total_compute / self.pp_degree
        bubble_ratio = bubble_time / pipeline_time if pipeline_time > 0 else 0

        # Baseline: all layers on one GPU, full batch
        baseline_ms = sum(p.avg_forward_ms for p in stage_profiles)

        result.total_time_ms = pipeline_time
        result.compute_time_ms = total_compute
        result.bubble_time_ms = bubble_time
        result.bubble_ratio = bubble_ratio
        result.baseline_time_ms = baseline_ms
        result.ideal_speedup = self.pp_degree
        result.actual_speedup = baseline_ms / pipeline_time if pipeline_time > 0 else 0
        result.pipeline_efficiency = result.actual_speedup / self.pp_degree if self.pp_degree > 0 else 0

        if pipeline_time > 0:
            result.throughput_samples_per_sec = (
                batch_size / (pipeline_time / 1000)
            )

        # Summary
        console.print(f"\n[bold]Pipeline Schedule (GPipe):[/]")
        console.print(f"  Stages: {self.pp_degree}, Micro-batches: {self.num_microbatches}")
        console.print(f"  Max stage time: {max_stage_ms:.2f} ms")
        console.print(f"  Pipeline time: {pipeline_time:.2f} ms")
        console.print(f"  Bubble time: {bubble_time:.2f} ms ({bubble_ratio:.1%})")
        console.print(f"  Speedup: {result.actual_speedup:.2f}x (ideal: {self.pp_degree}x)")
        console.print(f"  Efficiency: {result.pipeline_efficiency:.1%}")
        console.print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/s\n")

        return result

    def _profile_stage(
        self,
        layers: list[nn.Module],
        seq_len: int,
        micro_batch_size: int,
        warmup: int,
        iters: int,
    ) -> float:
        """Profile the forward time of a pipeline stage (group of layers)."""
        # Create dummy hidden states
        hidden_size = None
        for layer in layers:
            for p in layer.parameters():
                if p.dim() >= 2:
                    hidden_size = p.shape[-1]
                    break
            if hidden_size:
                break

        if hidden_size is None:
            return 0.0

        x = torch.randn(
            max(1, micro_batch_size), seq_len, hidden_size,
            dtype=self.dtype, device=self.device,
        )

        def run_stage():
            h = x
            for layer in layers:
                try:
                    out = layer(h)
                    # Handle tuple outputs (many transformer layers return tuples)
                    if isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out
                except Exception:
                    break
            return h

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                run_stage()
            torch.cuda.synchronize()

        # Time
        times = []
        with torch.no_grad():
            for _ in range(iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                run_stage()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        return sum(times) / len(times) if times else 0.0
