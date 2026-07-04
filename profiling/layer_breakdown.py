# TokenForge GPU-Accelerated LLM Inference Platform
"""
Per-layer timing breakdown.

Uses CUDA events to measure the execution time of each
individual transformer layer during inference. Helps identify
which layers are bottlenecks.
"""

import torch
from typing import Optional
from collections import OrderedDict

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config

console = Console()


def profile_layer_timing(
    model_name: Optional[str] = None,
    prompt: str = "Explain the attention mechanism in transformers.",
    num_runs: int = 5,
) -> dict:
    """
    Time each transformer layer individually using CUDA events.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model

    console.print(f"\n[bold]Layer Timing: {model_name}[/]\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Attach timing hooks to each named module
    layer_times: dict[str, list[float]] = OrderedDict()
    hooks = []

    def make_hooks(name):
        start_event = [None]
        end_event = [None]

        def pre_hook(module, input):
            start_event[0] = torch.cuda.Event(enable_timing=True)
            end_event[0] = torch.cuda.Event(enable_timing=True)
            start_event[0].record()

        def post_hook(module, input, output):
            end_event[0].record()
            torch.cuda.synchronize()
            elapsed = start_event[0].elapsed_time(end_event[0])
            if name not in layer_times:
                layer_times[name] = []
            layer_times[name].append(elapsed)

        return pre_hook, post_hook

    # Attach to transformer layers (look for common patterns)
    for name, module in model.named_modules():
        # Match transformer blocks / decoder layers
        parts = name.split(".")
        if any(
            p in ["h", "layers", "model.layers", "decoder.layers"]
            for p in [".".join(parts[:2]), ".".join(parts[:3])]
        ):
            # Only attach to top-level blocks, not sub-modules
            if len(parts) <= 3 and any(
                c.isdigit() for c in parts[-1:]
            ):
                pre, post = make_hooks(name)
                hooks.append(module.register_forward_pre_hook(pre))
                hooks.append(module.register_forward_hook(post))

    if not hooks:
        # Fallback: attach to all children of the main model body
        for name, module in model.named_children():
            for sub_name, sub_module in module.named_children():
                full_name = f"{name}.{sub_name}"
                if hasattr(sub_module, '__iter__') or 'layer' in sub_name.lower():
                    for idx, layer in enumerate(sub_module.children()):
                        layer_name = f"{full_name}.{idx}"
                        pre, post = make_hooks(layer_name)
                        hooks.append(layer.register_forward_pre_hook(pre))
                        hooks.append(layer.register_forward_hook(post))

    # Run inference
    for _ in range(num_runs + 1):  # +1 for warmup
        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate (skip first run = warmup)
    summary = {}
    for name, times in layer_times.items():
        run_times = times[1:]  # Skip warmup
        if run_times:
            avg_ms = sum(run_times) / len(run_times)
            summary[name] = avg_ms

    if summary:
        total_ms = sum(summary.values())

        table = Table(title="Per-Layer Timing (Forward Pass)")
        table.add_column("Layer", style="cyan")
        table.add_column("Avg Time (ms)", justify="right", style="green")
        table.add_column("% Total", justify="right", style="yellow")

        for name, avg_ms in summary.items():
            pct = (avg_ms / total_ms * 100) if total_ms > 0 else 0
            table.add_row(name, f"{avg_ms:.3f}", f"{pct:.1f}%")

        table.add_row("TOTAL", f"{total_ms:.3f}", "100.0%", style="bold")
        console.print(table)
    else:
        console.print("[yellow]No layer timing data collected. Model architecture not recognized.[/]")

    del model, tokenizer
    torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    profile_layer_timing(num_runs=3)
