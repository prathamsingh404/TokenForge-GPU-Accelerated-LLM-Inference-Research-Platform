"""
TokenForge CLI — Unified command-line interface.

Usage:
    tokenforge benchmark --model=gpt2 --scheduler=fifo --workload=chatgpt
    tokenforge profile   --model=gpt2 --depth=kernel
    tokenforge analyze   --experiment=<id>
    tokenforge compare   --experiments=exp1,exp2
    tokenforge serve     --model=gpt2 --port=8000
    tokenforge dashboard
    tokenforge info
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenforge",
        description=(
            "TokenForge — The open-source playground for studying, "
            "benchmarking, profiling, and optimizing LLM inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tokenforge benchmark --model gpt2\n"
            "  tokenforge benchmark --model gpt2 --scheduler priority --workload chatgpt\n"
            "  tokenforge profile --model gpt2 --depth kernel\n"
            "  tokenforge analyze\n"
            "  tokenforge compare --experiments exp1 exp2\n"
            "  tokenforge dashboard\n"
            "  tokenforge info\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- benchmark ---
    bench = subparsers.add_parser(
        "benchmark",
        help="Run inference benchmarks with configurable workloads and schedulers",
    )
    bench.add_argument("--model", type=str, default=None,
                       help="HuggingFace model ID (default: gpt2)")
    bench.add_argument("--quantization", type=str, default="fp16",
                       choices=["fp32", "fp16", "bf16", "int8", "int4"],
                       help="Precision format (default: fp16)")
    bench.add_argument("--scheduler", type=str, default="fifo",
                       choices=["fifo", "round_robin", "continuous", "priority",
                                "shortest_remaining", "deadline_aware", "token_fair"],
                       help="Scheduling algorithm (default: fifo)")
    bench.add_argument("--workload", type=str, default=None,
                       choices=["chatgpt", "rag", "coding", "customer_support"],
                       help="Pre-built workload profile (default: single prompt)")
    bench.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for single-prompt mode (default: 1)")
    bench.add_argument("--max-tokens", type=int, default=128,
                       help="Max new tokens per request (default: 128)")
    bench.add_argument("--num-runs", type=int, default=10,
                       help="Number of timed iterations (default: 10)")
    bench.add_argument("--duration", type=int, default=60,
                       help="Workload simulation duration in seconds (default: 60)")
    bench.add_argument("--num-users", type=int, default=100,
                       help="Simulated concurrent users for workload mode (default: 100)")
    bench.add_argument("--prompt", type=str, default=None,
                       help="Custom prompt for single-prompt mode")
    bench.add_argument("--output", type=str, default=None,
                       help="Output directory for results (default: experiments/)")
    bench.add_argument("--no-db", action="store_true",
                       help="Skip saving results to SQLite database")
    bench.add_argument("--json-only", action="store_true",
                       help="Output JSON results to stdout and exit")

    # --- profile ---
    prof = subparsers.add_parser(
        "profile",
        help="Profile model inference at kernel, layer, or pipeline level",
    )
    prof.add_argument("--model", type=str, default=None,
                      help="HuggingFace model ID (default: gpt2)")
    prof.add_argument("--depth", type=str, default="pipeline",
                      choices=["pipeline", "layer", "kernel"],
                      help="Profiling depth (default: pipeline)")
    prof.add_argument("--max-tokens", type=int, default=16,
                      help="Tokens to generate during profiling (default: 16)")
    prof.add_argument("--output", type=str, default=None,
                      help="Output directory for traces")

    # --- analyze ---
    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze results and generate optimization recommendations",
    )
    analyze.add_argument("--experiment", type=str, default=None,
                         help="Experiment ID to analyze (default: latest)")
    analyze.add_argument("--model", type=str, default=None,
                         help="Model to run quick analysis on")

    # --- compare ---
    compare = subparsers.add_parser(
        "compare",
        help="Compare results across multiple experiments",
    )
    compare.add_argument("--experiments", nargs="+", required=True,
                         help="Experiment IDs to compare")
    compare.add_argument("--metric", type=str, default="throughput_avg",
                         help="Primary metric to compare (default: throughput_avg)")

    # --- serve ---
    serve = subparsers.add_parser(
        "serve",
        help="Launch the inference server with a loaded model",
    )
    serve.add_argument("--model", type=str, default=None,
                       help="HuggingFace model ID")
    serve.add_argument("--quantization", type=str, default="fp16",
                       choices=["fp32", "fp16", "bf16", "int8", "int4"])
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", type=str, default="0.0.0.0")

    # --- dashboard ---
    dash = subparsers.add_parser(
        "dashboard",
        help="Launch the web dashboard",
    )
    dash.add_argument("--port", type=int, default=8000)
    dash.add_argument("--host", type=str, default="0.0.0.0")

    # --- info ---
    subparsers.add_parser(
        "info",
        help="Show system information, GPU details, and installed capabilities",
    )

    return parser


def _get_version() -> str:
    try:
        from tokenforge import __version__
        return __version__
    except ImportError:
        return "0.2.0"


# ─── Command Handlers ────────────────────────────────────────────────

def cmd_benchmark(args):
    """Execute a benchmark run."""
    from rich.console import Console
    console = Console()

    console.print(f"\n[bold cyan]{'━' * 60}[/]")
    console.print(f"[bold]TokenForge Benchmark[/]")
    console.print(f"[bold cyan]{'━' * 60}[/]\n")

    if args.workload:
        # Workload simulation mode
        from tokenforge.workloads.generator import WorkloadConfig

        presets = {
            "chatgpt": WorkloadConfig.chatgpt_traffic,
            "rag": WorkloadConfig.rag_traffic,
            "coding": WorkloadConfig.coding_assistant,
            "customer_support": WorkloadConfig.customer_support,
        }

        workload = presets[args.workload]()
        workload.duration_seconds = args.duration
        workload.num_users = args.num_users

        console.print(f"[bold]Workload:[/] {args.workload}")
        console.print(f"[bold]Users:[/] {workload.num_users}")
        console.print(f"[bold]Duration:[/] {workload.duration_seconds}s")
        console.print(f"[bold]Arrival:[/] {workload.arrival_pattern}")
        console.print(f"[bold]Scheduler:[/] {args.scheduler}")
        console.print(f"[bold]Model:[/] {args.model or 'default'}\n")

        # Generate the workload requests
        requests = workload.generate_requests()
        console.print(f"[dim]Generated {len(requests)} simulated requests[/]\n")

        # TODO: Integrate with scheduler framework + engine for full workload run
        # For now, run individual requests through benchmark engine
        console.print("[yellow]Full workload simulation coming in Phase 2 integration.[/]")
        console.print(f"[dim]Workload config: {workload}[/]")

    else:
        # Single-prompt benchmark mode (existing behavior, enhanced)
        from core.config import get_config
        cfg = get_config()

        model_name = args.model or cfg.models.medium_model
        prompt = args.prompt or "Explain the concept of attention mechanisms in transformer architectures."

        console.print(f"[bold]Model:[/] {model_name}")
        console.print(f"[bold]Quantization:[/] {args.quantization}")
        console.print(f"[bold]Batch Size:[/] {args.batch_size}")
        console.print(f"[bold]Runs:[/] {args.num_runs}\n")

        from benchmark_engine.inference_bench import run_inference_benchmark
        result = run_inference_benchmark(
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
            num_runs=args.num_runs,
            quantization=args.quantization,
        )

        # Attach environment manifest
        from tokenforge.environment import EnvironmentManifest
        manifest = EnvironmentManifest.capture(
            model_name=model_name,
            quantization=args.quantization,
        )
        console.print(f"\n[dim]Environment manifest captured: {manifest.short_summary()}[/]")

        if args.json_only:
            print(json.dumps(result.to_dict(), indent=2, default=str))


def cmd_profile(args):
    """Run profiling at the requested depth."""
    from rich.console import Console
    console = Console()

    console.print(f"\n[bold cyan]TokenForge Profiler[/] — depth: {args.depth}\n")

    if args.depth == "pipeline":
        from benchmark_engine.pipeline_profiler import profile_pipeline
        profile_pipeline(
            model_name=args.model,
            max_new_tokens=args.max_tokens,
        )
    elif args.depth == "layer":
        from profiling.layer_breakdown import profile_layer_timing
        profile_layer_timing(model_name=args.model)
    elif args.depth == "kernel":
        from profiling.torch_profiler import profile_inference
        profile_inference(
            model_name=args.model,
            max_new_tokens=args.max_tokens,
        )


def cmd_analyze(args):
    """Run automated optimization analysis."""
    from rich.console import Console
    console = Console()

    console.print(f"\n[bold cyan]TokenForge Analyzer[/]\n")

    try:
        from tokenforge.analyze import run_analysis
        run_analysis(
            experiment_id=args.experiment,
            model_name=args.model,
        )
    except ImportError:
        console.print("[yellow]Analysis module will be available in Phase 4.[/]")
        console.print("[dim]For now, use 'tokenforge profile' for performance insights.[/]")


def cmd_compare(args):
    """Compare multiple experiments side-by-side."""
    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print(f"\n[bold cyan]TokenForge Experiment Comparison[/]\n")

    from core.database import ExperimentDB

    with ExperimentDB() as db:
        table = Table(title="Experiment Comparison", show_header=True)
        table.add_column("Experiment", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Quantization", style="yellow")
        table.add_column(args.metric, justify="right", style="green")

        for exp_id in args.experiments:
            exp = db.get_experiment(exp_id)
            if not exp:
                console.print(f"[red]Experiment '{exp_id}' not found[/]")
                continue

            metrics = db.get_metrics(exp_id)
            target = next(
                (m for m in metrics if m["metric_name"] == args.metric),
                None,
            )
            value = f"{target['metric_value']:.2f} {target['unit']}" if target else "—"

            table.add_row(
                exp["name"],
                exp["model_name"],
                exp.get("quantization", "—"),
                value,
            )

        console.print(table)


def cmd_serve(args):
    """Launch the inference server."""
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold cyan]TokenForge Inference Server[/]")
    console.print(f"[dim]Model: {args.model or 'default'} | Port: {args.port}[/]\n")

    import uvicorn
    from dashboard.app import app
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_dashboard(args):
    """Launch the web dashboard."""
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold cyan]TokenForge Dashboard[/]")
    console.print(f"[dim]http://{args.host}:{args.port}[/]\n")

    import uvicorn
    from dashboard.app import app
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_info(args):
    """Display system information."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    from tokenforge.environment import EnvironmentManifest
    manifest = EnvironmentManifest.capture()

    panel_content = (
        f"[bold]TokenForge[/] v{_get_version()}\n"
        f"\n"
        f"[cyan]GPU:[/]          {manifest.gpu_name}\n"
        f"[cyan]VRAM:[/]         {manifest.gpu_vram_mb} MB\n"
        f"[cyan]CUDA:[/]         {manifest.cuda_version}\n"
        f"[cyan]Driver:[/]       {manifest.gpu_driver_version}\n"
        f"[cyan]PyTorch:[/]      {manifest.pytorch_version}\n"
        f"[cyan]Python:[/]       {manifest.python_version}\n"
        f"[cyan]OS:[/]           {manifest.os_info}\n"
        f"[cyan]Git Commit:[/]   {manifest.git_commit_hash or 'N/A'}\n"
        f"[cyan]Timestamp:[/]    {manifest.timestamp}\n"
    )

    console.print(Panel(panel_content, title="System Information", border_style="cyan"))

    # List available schedulers
    try:
        from tokenforge.schedulers import list_schedulers
        schedulers = list_schedulers()
        console.print(f"\n[bold]Available Schedulers:[/] {', '.join(schedulers)}")
    except ImportError:
        pass

    # List available workloads
    try:
        from tokenforge.workloads.presets import WORKLOAD_PRESETS
        console.print(f"[bold]Workload Presets:[/] {', '.join(WORKLOAD_PRESETS.keys())}")
    except ImportError:
        pass


# ─── Main Entry Point ────────────────────────────────────────────────

COMMANDS = {
    "benchmark": cmd_benchmark,
    "profile": cmd_profile,
    "analyze": cmd_analyze,
    "compare": cmd_compare,
    "serve": cmd_serve,
    "dashboard": cmd_dashboard,
    "info": cmd_info,
}


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = COMMANDS.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(130)
        except Exception as e:
            from rich.console import Console
            Console().print(f"\n[bold red]Error:[/] {e}")
            raise
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
