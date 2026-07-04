"""
KV Cache utilization visualization.

Plots cache utilization over time across different tiers (GPU, CPU).
Useful for understanding memory fragmentation, eviction rates, and
tier movement in hierarchical setups.

Produces an interactive HTML plot or static matplotlib figure.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class CacheSnapshot:
    """A single point in time for cache stats."""
    timestamp_s: float
    gpu_used_mb: float
    gpu_budget_mb: float
    cpu_used_mb: float
    cpu_budget_mb: float
    active_requests: int


class CacheUtilizationPlot:
    """
    Visualizes KV cache memory usage over time.

    Shows GPU and CPU tier utilization, helping identify memory
    bottlenecks, fragmentation, and eviction patterns during
    long-running simulations.
    """

    def __init__(self):
        self._snapshots: list[CacheSnapshot] = []

    def add_snapshot(self, snapshot: CacheSnapshot):
        self._snapshots.append(snapshot)

    def render_matplotlib(
        self,
        output_path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 6),
    ):
        """Render utilization timeline as a matplotlib plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required. Install with: pip install matplotlib")
            return

        if not self._snapshots:
            return

        times = [s.timestamp_s for s in self._snapshots]
        gpu_used = [s.gpu_used_mb for s in self._snapshots]
        gpu_budget = [s.gpu_budget_mb for s in self._snapshots]
        
        has_cpu = any(s.cpu_budget_mb > 0 for s in self._snapshots)
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # GPU Tier
        ax1.plot(times, gpu_used, color="#10b981", linewidth=2, label="GPU Used (MB)")
        ax1.plot(times, gpu_budget, color="#10b981", linestyle="--", alpha=0.5, label="GPU Budget")
        ax1.fill_between(times, 0, gpu_used, color="#10b981", alpha=0.2)
        
        # CPU Tier
        if has_cpu:
            cpu_used = [s.cpu_used_mb for s in self._snapshots]
            cpu_budget = [s.cpu_budget_mb for s in self._snapshots]
            ax1.plot(times, cpu_used, color="#6366f1", linewidth=2, label="CPU Used (MB)")
            ax1.plot(times, cpu_budget, color="#6366f1", linestyle="--", alpha=0.5, label="CPU Budget")
            ax1.fill_between(times, 0, cpu_used, color="#6366f1", alpha=0.1)

        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylabel("Memory (MB)", fontsize=12)
        ax1.set_title("Hierarchical KV Cache Utilization", fontsize=14, fontweight="bold")
        
        # Secondary axis for active requests
        ax2 = ax1.twinx()
        reqs = [s.active_requests for s in self._snapshots]
        ax2.plot(times, reqs, color="#f59e0b", linewidth=1, alpha=0.7, label="Active Requests")
        ax2.set_ylabel("Active Requests", fontsize=12, color="#f59e0b")
        ax2.tick_params(axis='y', labelcolor="#f59e0b")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {output_path}")

        plt.close(fig)
        return fig
