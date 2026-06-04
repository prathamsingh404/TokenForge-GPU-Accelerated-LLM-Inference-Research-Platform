"""
Report generator for benchmark experiments.

Reads experiment data from the database and generates formatted
reports with tables, comparisons, and chart data.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table

from core.config import get_config
from core.database import ExperimentDB

console = Console()


def generate_phase_report(phase: str, output_dir: Optional[Path] = None) -> dict:
    """
    Generate a summary report for all experiments in a given phase.
    Returns the report data and saves it as JSON.
    """
    cfg = get_config()
    output_dir = output_dir or cfg.reports_dir

    with ExperimentDB() as db:
        experiments = db.list_experiments(phase=phase)
        if not experiments:
            console.print(f"[yellow]No experiments found for phase: {phase}[/]")
            return {}

        report = {
            "phase": phase,
            "generated_at": datetime.now().isoformat(),
            "gpu": cfg.hardware.gpu_name,
            "experiment_count": len(experiments),
            "experiments": [],
        }

        for exp in experiments:
            metrics = db.get_metrics(exp["id"])
            gpu_snaps = db.get_gpu_snapshots(exp["id"])

            exp_data = {
                "id": exp["id"],
                "name": exp["name"],
                "model": exp["model_name"],
                "batch_size": exp["batch_size"],
                "quantization": exp["quantization"],
                "metrics": {m["metric_name"]: m["metric_value"] for m in metrics},
                "gpu_samples": len(gpu_snaps),
            }
            report["experiments"].append(exp_data)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{phase}_{int(datetime.now().timestamp())}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[green]Report saved to {report_path}[/]")
    return report


def print_comparison_table(phase: str, metric_name: str = "throughput_avg"):
    """Print a side-by-side comparison of a specific metric across experiments."""
    with ExperimentDB() as db:
        data = db.get_comparison_data(phase, metric_name)

    if not data:
        console.print(f"[yellow]No data for {metric_name} in phase {phase}[/]")
        return

    table = Table(title=f"{phase.title()} — {metric_name}", show_header=True)
    table.add_column("Experiment", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Batch Size", justify="right")
    table.add_column("Quantization", style="magenta")
    table.add_column(metric_name, justify="right", style="green")
    table.add_column("Unit")

    for row in data:
        table.add_row(
            row["name"],
            row["model_name"],
            str(row["batch_size"] or "—"),
            row["quantization"] or "—",
            f"{row['metric_value']:.2f}",
            row["unit"],
        )

    console.print(table)


def list_all_experiments():
    """Print a summary of all recorded experiments."""
    with ExperimentDB() as db:
        experiments = db.list_experiments(limit=200)

    if not experiments:
        console.print("[yellow]No experiments in database.[/]")
        return

    table = Table(title="All Experiments", show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Phase", style="magenta")
    table.add_column("Model")
    table.add_column("BS", justify="right")
    table.add_column("Quant")
    table.add_column("Created", style="dim")

    for exp in experiments:
        table.add_row(
            exp["id"][:8],
            exp["name"],
            exp["phase"],
            exp["model_name"].split("/")[-1],
            str(exp["batch_size"] or "—"),
            exp["quantization"] or "—",
            exp["created_at"][:19],
        )

    console.print(table)


if __name__ == "__main__":
    list_all_experiments()
