"""
Expert Parallelism simulation for Mixture-of-Experts (MoE) models.

Simulates the distribution of experts across multiple GPUs and
analyzes expert routing patterns, load balancing, and communication
overhead from all-to-all exchange.

Key concepts:
- Top-K gating: Each token is routed to K experts out of N total.
- All-to-all: Tokens must be shuffled across GPUs to reach their
  assigned experts, then shuffled back after computation.
- Load imbalance: Some experts receive more tokens than others,
  causing stragglers.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertProfile:
    """Profile of a single expert."""
    expert_id: int
    gpu_id: int
    tokens_received: int = 0
    compute_time_ms: float = 0.0
    load_fraction: float = 0.0


@dataclass
class EPSimulationResult:
    """Results from an expert parallelism simulation."""
    num_experts: int
    experts_per_token: int
    ep_degree: int
    total_tokens: int

    # Expert-level metrics
    expert_profiles: list[ExpertProfile] = field(default_factory=list)

    # Load balancing
    load_imbalance_ratio: float = 0.0  # max/avg token count
    coefficient_of_variation: float = 0.0

    # Communication
    all_to_all_bytes: int = 0
    all_to_all_time_ms: float = 0.0

    # Timing
    compute_time_ms: float = 0.0
    total_time_ms: float = 0.0


class ExpertParallelSimulator:
    """
    Simulate expert parallelism for MoE models.

    Analyzes expert routing patterns and load distribution.
    Does not require an actual MoE model — can simulate routing
    with configurable parameters.

    Usage:
        sim = ExpertParallelSimulator(
            num_experts=64, experts_per_token=2, ep_degree=8,
        )
        result = sim.run(seq_len=2048, batch_size=4)
    """

    def __init__(
        self,
        num_experts: int = 8,
        experts_per_token: int = 2,
        ep_degree: int = 2,
        expert_hidden_size: int = 4096,
        expert_intermediate_size: int = 14336,
        inter_gpu_bandwidth_gbps: float = 600.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.ep_degree = ep_degree
        self.expert_hidden = expert_hidden_size
        self.expert_intermediate = expert_intermediate_size
        self.bandwidth_gbps = inter_gpu_bandwidth_gbps
        self.device = device
        self.dtype = dtype

    def _simulate_gating(
        self, total_tokens: int, seed: int = 42,
    ) -> torch.Tensor:
        """
        Simulate top-K gating decisions.

        Returns expert assignments: [total_tokens, experts_per_token]
        """
        torch.manual_seed(seed)

        # Simulate gate logits (random routing)
        logits = torch.randn(total_tokens, self.num_experts, device=self.device)
        _, indices = logits.topk(self.experts_per_token, dim=-1)
        return indices

    def run(
        self,
        seq_len: int = 2048,
        batch_size: int = 4,
        warmup_iters: int = 3,
        timed_iters: int = 5,
    ) -> EPSimulationResult:
        """
        Run expert parallelism simulation.

        1. Simulate gating (top-K routing)
        2. Analyze load distribution across experts
        3. Estimate all-to-all communication cost
        4. Profile expert compute time
        """
        from rich.console import Console
        from rich.table import Table
        console = Console()

        console.print(f"\n[bold cyan]Expert Parallelism Simulation[/]")
        console.print(f"  Experts: {self.num_experts}, Top-K: {self.experts_per_token}, "
                       f"EP: {self.ep_degree}\n")

        total_tokens = batch_size * seq_len
        assignments = self._simulate_gating(total_tokens)

        result = EPSimulationResult(
            num_experts=self.num_experts,
            experts_per_token=self.experts_per_token,
            ep_degree=self.ep_degree,
            total_tokens=total_tokens,
        )

        # Count tokens per expert
        expert_counts = torch.zeros(self.num_experts, dtype=torch.int64, device=self.device)
        for k in range(self.experts_per_token):
            expert_ids = assignments[:, k]
            expert_counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))

        expert_counts = expert_counts.cpu().tolist()

        # Expert-to-GPU mapping
        experts_per_gpu = self.num_experts // self.ep_degree

        profiles = []
        for eid in range(self.num_experts):
            gpu_id = eid // experts_per_gpu
            profiles.append(ExpertProfile(
                expert_id=eid,
                gpu_id=gpu_id,
                tokens_received=expert_counts[eid],
                load_fraction=expert_counts[eid] / (total_tokens * self.experts_per_token),
            ))

        result.expert_profiles = profiles

        # Load balancing analysis
        counts = torch.tensor([p.tokens_received for p in profiles], dtype=torch.float32)
        avg_count = counts.mean().item()
        max_count = counts.max().item()

        result.load_imbalance_ratio = max_count / avg_count if avg_count > 0 else 0
        std_count = counts.std().item()
        result.coefficient_of_variation = std_count / avg_count if avg_count > 0 else 0

        # Display load distribution
        table = Table(title="Expert Load Distribution")
        table.add_column("Expert", justify="center", style="cyan")
        table.add_column("GPU", justify="center")
        table.add_column("Tokens", justify="right", style="green")
        table.add_column("Load %", justify="right", style="yellow")
        table.add_column("Distribution", justify="left")

        for p in profiles:
            bar_len = int(30 * p.tokens_received / max_count) if max_count > 0 else 0
            bar = "█" * bar_len
            table.add_row(
                str(p.expert_id),
                str(p.gpu_id),
                str(p.tokens_received),
                f"{p.load_fraction:.1%}",
                bar,
            )

        console.print(table)

        # Communication: all-to-all exchange
        # Each token that crosses GPU boundaries needs to be transferred
        cross_gpu_tokens = 0
        for eid, count in enumerate(expert_counts):
            expert_gpu = eid // experts_per_gpu
            # Assume tokens are uniformly distributed across GPUs
            local_fraction = 1.0 / self.ep_degree
            remote_tokens = int(count * (1 - local_fraction))
            cross_gpu_tokens += remote_tokens

        # Bytes: each token sends hidden_size activations in FP16
        a2a_bytes = cross_gpu_tokens * self.expert_hidden * 2  # FP16
        bandwidth_bytes_ms = self.bandwidth_gbps * 1e9 / 8 / 1000
        a2a_time_ms = a2a_bytes / bandwidth_bytes_ms

        result.all_to_all_bytes = a2a_bytes
        result.all_to_all_time_ms = a2a_time_ms

        # Profile expert compute
        expert_time = self._profile_expert(warmup_iters, timed_iters, int(avg_count))
        result.compute_time_ms = expert_time * self.num_experts / self.ep_degree

        # Straggler effect: total time limited by slowest GPU
        gpu_loads = {}
        for p in profiles:
            gpu_loads[p.gpu_id] = gpu_loads.get(p.gpu_id, 0) + p.tokens_received

        max_gpu_tokens = max(gpu_loads.values()) if gpu_loads else 0
        straggler_factor = max_gpu_tokens / avg_count if avg_count > 0 else 1.0

        result.total_time_ms = result.compute_time_ms * straggler_factor + a2a_time_ms

        # Summary
        console.print(f"\n[bold]Analysis:[/]")
        console.print(f"  Load Imbalance: {result.load_imbalance_ratio:.2f}x "
                       f"(1.0 = perfectly balanced)")
        console.print(f"  CV: {result.coefficient_of_variation:.3f}")
        console.print(f"  Cross-GPU tokens: {cross_gpu_tokens:,} "
                       f"({a2a_bytes / (1024**2):.1f} MB all-to-all)")
        console.print(f"  Communication: {a2a_time_ms:.2f} ms")
        console.print(f"  Compute: {result.compute_time_ms:.2f} ms")
        console.print(f"  Straggler factor: {straggler_factor:.2f}x")
        console.print(f"  Total: {result.total_time_ms:.2f} ms\n")

        return result

    def _profile_expert(
        self, warmup: int, iters: int, tokens_per_expert: int,
    ) -> float:
        """Profile a single expert's forward pass time."""
        if tokens_per_expert <= 0:
            return 0.0

        # Simulate expert FFN: gate_proj → up_proj → down_proj (SwiGLU)
        x = torch.randn(
            tokens_per_expert, self.expert_hidden,
            dtype=self.dtype, device=self.device,
        )
        gate = torch.randn(
            self.expert_intermediate, self.expert_hidden,
            dtype=self.dtype, device=self.device,
        )
        up = torch.randn(
            self.expert_intermediate, self.expert_hidden,
            dtype=self.dtype, device=self.device,
        )
        down = torch.randn(
            self.expert_hidden, self.expert_intermediate,
            dtype=self.dtype, device=self.device,
        )

        def run():
            g = F.silu(x @ gate.t())
            u = x @ up.t()
            return (g * u) @ down.t()

        with torch.no_grad():
            for _ in range(warmup):
                run()
            torch.cuda.synchronize()

            times = []
            for _ in range(iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                run()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        return sum(times) / len(times) if times else 0.0
