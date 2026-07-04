"""
Latency/throughput heatmaps.

Generates 2D heatmaps showing how performance varies across
two dimensions (batch size × sequence length, prompt length ×
output length, etc.). Produces publication-quality matplotlib
figures.

Usage:
    from tokenforge.visualization.heatmaps import LatencyHeatmap

    heatmap = LatencyHeatmap()
    heatmap.generate(results_matrix, x_axis="batch_size", y_axis="seq_len")
    heatmap.save("latency_heatmap.png")
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    title: str = "Latency Heatmap"
    x_label: str = "Batch Size"
    y_label: str = "Sequence Length"
    value_label: str = "Latency (ms)"
    colormap: str = "RdYlGn_r"  # Red=slow, Green=fast
    figsize: tuple[int, int] = (12, 8)
    annotate: bool = True
    dpi: int = 150


class LatencyHeatmap:
    """
    2D heatmap for performance metrics.

    Visualizes how latency/throughput/memory changes across
    two parameter dimensions.
    """

    def __init__(self, config: Optional[HeatmapConfig] = None):
        self.config = config or HeatmapConfig()

    def generate(
        self,
        data: dict,
        x_values: list,
        y_values: list,
        output_path: Optional[str] = None,
    ):
        """
        Generate a heatmap from a 2D data grid.

        Args:
            data: Dict mapping (x, y) tuples to metric values.
            x_values: Sorted list of x-axis values.
            y_values: Sorted list of y-axis values.
            output_path: File path to save figure (optional).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required for heatmaps. Install with: pip install matplotlib")
            return

        # Build matrix
        matrix = np.zeros((len(y_values), len(x_values)))
        for yi, y in enumerate(y_values):
            for xi, x in enumerate(x_values):
                matrix[yi, xi] = data.get((x, y), 0.0)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        im = ax.imshow(
            matrix, cmap=self.config.colormap, aspect="auto",
            interpolation="nearest",
        )

        # Axis labels
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values], rotation=45)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([str(v) for v in y_values])
        ax.set_xlabel(self.config.x_label, fontsize=12)
        ax.set_ylabel(self.config.y_label, fontsize=12)
        ax.set_title(self.config.title, fontsize=14, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label=self.config.value_label)

        # Annotate cells
        if self.config.annotate:
            for yi in range(len(y_values)):
                for xi in range(len(x_values)):
                    val = matrix[yi, xi]
                    color = "white" if val > matrix.mean() else "black"
                    ax.text(xi, yi, f"{val:.1f}",
                           ha="center", va="center", color=color, fontsize=8)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Heatmap saved to {output_path}")

        plt.close(fig)
        return fig


class ThroughputHeatmap(LatencyHeatmap):
    """Throughput-specific heatmap (higher = greener)."""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        config = config or HeatmapConfig(
            title="Throughput Heatmap",
            value_label="Tokens/s",
            colormap="RdYlGn",  # Green=fast (reversed)
        )
        super().__init__(config)


class MemoryHeatmap(LatencyHeatmap):
    """Memory usage heatmap."""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        config = config or HeatmapConfig(
            title="VRAM Usage Heatmap",
            value_label="VRAM (MB)",
            colormap="YlOrRd",
        )
        super().__init__(config)
