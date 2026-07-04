"""
Interactive Roofline Plot visualization.

The Roofline model visually relates performance (TFLOPS) and
memory bandwidth (GB/s) to arithmetic intensity (FLOPs/byte).
It helps identify whether a kernel is compute-bound or
memory-bound.

This module upgrades the existing roofline logic to output
high-quality visualization.
"""

from typing import Optional
import math


class RooflinePlot:
    """
    Visualizes kernel performance against hardware limits.
    """

    def __init__(
        self,
        peak_tflops: float,
        peak_bandwidth_gbps: float,
    ):
        self.peak_tflops = peak_tflops
        self.peak_bandwidth_gbps = peak_bandwidth_gbps
        # Ridge point: TFLOPS / Bandwidth
        self.ridge_point = (peak_tflops * 1000) / peak_bandwidth_gbps

    def render_matplotlib(
        self,
        kernels: list[dict],
        output_path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
    ):
        """
        Render a roofline plot.
        
        Args:
            kernels: List of dicts with 'name', 'intensity' (FLOPs/byte),
                     and 'performance' (TFLOPS).
            output_path: Where to save the image.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required. Install with: pip install matplotlib")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Plot the "roofs"
        # X axis: Arithmetic Intensity (FLOPs/byte)
        # Y axis: Performance (TFLOPS)
        x = np.logspace(-2, 4, 1000)
        
        # Memory bandwidth roof: y = x * bandwidth
        y_mem = x * (self.peak_bandwidth_gbps / 1000)
        
        # Compute roof: y = peak_tflops
        y_comp = np.full_like(x, self.peak_tflops)
        
        # The actual roof is the minimum of the two
        y_roof = np.minimum(y_mem, y_comp)

        ax.plot(x, y_roof, color="red", linewidth=2, label="Hardware Limit")
        
        # Plot kernels
        for kernel in kernels:
            intensity = kernel.get("intensity", 0)
            perf = kernel.get("performance", 0)
            name = kernel.get("name", "Unknown")
            
            # Determine bound
            color = "#10b981" if intensity >= self.ridge_point else "#6366f1"
            
            ax.scatter([intensity], [perf], color=color, s=100, zorder=5, edgecolor="white")
            ax.annotate(
                name,
                (intensity, perf),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        
        ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=12)
        ax.set_ylabel("Performance (TFLOPS)", fontsize=12)
        ax.set_title("Roofline Analysis", fontsize=14, fontweight="bold")
        
        ax.grid(True, which="both", ls="--", alpha=0.2)
        ax.legend()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Roofline plot saved to {output_path}")

        plt.close(fig)
        return fig
