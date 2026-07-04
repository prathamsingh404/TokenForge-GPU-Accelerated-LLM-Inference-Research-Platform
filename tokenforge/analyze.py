"""
Automated optimization recommendations.

Analyzes experimental results and hardware constraints to provide
actionable recommendations for improving inference performance.

Usage:
    from tokenforge.analyze import run_analysis
    
    run_analysis(experiment_id="exp_001")
"""

import sys
from typing import Optional
from dataclasses import dataclass

from core.database import ExperimentDB
from tokenforge.model_registry import ArchitectureInfo, detect_architecture


@dataclass
class Recommendation:
    title: str
    description: str
    expected_impact: str
    difficulty: str  # low, medium, high


def run_analysis(experiment_id: Optional[str] = None, model_name: Optional[str] = None):
    """
    Analyze an experiment and generate optimization recommendations.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()

    console.print("\n[bold cyan]TokenForge Optimizer[/] — Automated Analysis\n")

    db = ExperimentDB()
    
    if experiment_id:
        exp = db.get_experiment(experiment_id)
        if not exp:
            console.print(f"[bold red]Error:[/] Experiment {experiment_id} not found.")
            return
        
        metrics = db.get_metrics(experiment_id)
        model_name = exp.get("model_name", "Unknown")
        quantization = exp.get("quantization", "fp16")
    else:
        # Fallback to general model analysis if no experiment specified
        exp = None
        metrics = []
        quantization = "fp16"

    if not model_name:
        console.print("[yellow]Please provide an experiment ID or model name.[/]")
        return

    # Mock architecture info for analysis since we don't load the model here
    # In a full implementation, we'd fetch this from HF config
    console.print(f"[dim]Analyzing target model: {model_name}[/]")
    
    recommendations = []

    # 1. Batching Analysis
    if exp:
        batch_size = exp.get("batch_size", 1)
        if batch_size == 1:
            recommendations.append(Recommendation(
                title="Enable Continuous Batching",
                description="Current run uses batch_size=1. For serving workloads, continuous iteration-level batching dramatically improves throughput.",
                expected_impact="High (2x-5x throughput increase)",
                difficulty="low"
            ))

    # 2. Quantization Analysis
    if quantization in ["fp32", "fp16"]:
        recommendations.append(Recommendation(
            title="Switch to INT8/INT4 Quantization",
            description=f"Model is running in {quantization}. Quantizing to INT8/INT4 reduces VRAM usage and increases memory bandwidth utilization.",
            expected_impact="High (~50% VRAM reduction, 1.2x throughput)",
            difficulty="low"
        ))

    # 3. KV Cache Analysis
    if metrics:
        vram_metrics = [m for m in metrics if "vram" in m["metric_name"].lower()]
        peak_vram = max([m["metric_value"] for m in vram_metrics]) if vram_metrics else 0
        if peak_vram > 16000:  # Arbitrary threshold
            recommendations.append(Recommendation(
                title="Hierarchical KV Cache",
                description="High VRAM usage detected. Enable GPU->CPU KV cache offloading to support longer contexts without OOM.",
                expected_impact="Medium (Supports 4x larger context lengths)",
                difficulty="medium"
            ))

    # 4. Scheduling Analysis
    recommendations.append(Recommendation(
        title="Token-Fair or Priority Scheduling",
        description="Standard FIFO can starve short requests behind long ones. Use SRPT or Token-Fair scheduling for better P99 tail latency.",
        expected_impact="Medium (Significantly better P99 latency)",
        difficulty="low"
    ))
    
    # 5. Distributed Analysis
    recommendations.append(Recommendation(
        title="Tensor Parallelism",
        description="To further reduce latency for single requests, split weights across GPUs using Tensor Parallelism.",
        expected_impact="High (Near-linear latency reduction with GPU count)",
        difficulty="high"
    ))

    if not recommendations:
        console.print("[green]System looks well optimized! No critical recommendations found.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Recommendation", style="cyan", width=30)
    table.add_column("Description", style="white")
    table.add_column("Impact", style="green")

    for rec in recommendations:
        table.add_row(rec.title, rec.description, rec.expected_impact)

    console.print(table)
    console.print("\n[dim]Run `tokenforge benchmark` with new flags to apply these changes.[/]\n")
