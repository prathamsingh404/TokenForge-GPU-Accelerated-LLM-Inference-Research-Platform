# TokenForge GPU-Accelerated LLM Inference Platform
"""
Per-stage pipeline profiler.

Instruments a transformer model to measure time spent in each
stage of the inference pipeline: tokenization, embedding lookup,
attention, feed-forward, layer norm, and sampling.
"""

import time
from collections import defaultdict
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table

from core.config import get_config

console = Console()


class LayerTimer:
    """
    Attaches forward hooks to model layers to measure execution time.
    Uses CUDA events for accurate GPU timing.
    """

    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._hooks = []
        self._cuda_events: dict[str, tuple] = {}

    def _make_hook(self, name: str):
        def pre_hook(module, input):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._cuda_events[name] = (start, end)

        def post_hook(module, input, output):
            if name in self._cuda_events:
                start, end = self._cuda_events[name]
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                self.timings[name].append(elapsed_ms)

        return pre_hook, post_hook

    def attach(self, model):
        """Walk the model and attach timing hooks to key layer types."""
        layer_types = {
            "Embedding": (torch.nn.Embedding,),
            "LayerNorm": (torch.nn.LayerNorm,),
            "Linear": (torch.nn.Linear,),
        }

        # Also look for attention and MLP modules by name
        for name, module in model.named_modules():
            label = None

            if "attn" in name.lower() or "attention" in name.lower():
                if not any(c for c in module.children()):
                    continue  # Skip leaf modules inside attention
                label = f"Attention"
            elif "mlp" in name.lower() or "ffn" in name.lower():
                if not any(c for c in module.children()):
                    continue
                label = f"FFN/MLP"
            elif isinstance(module, torch.nn.Embedding):
                label = "Embedding"
            elif isinstance(module, torch.nn.LayerNorm):
                label = "LayerNorm"

            if label:
                # Use unique names for multiple instances
                idx = sum(1 for k in self.timings if k.startswith(label))
                unique_name = f"{label}_{idx}" if idx > 0 else label
                pre, post = self._make_hook(unique_name)
                h1 = module.register_forward_pre_hook(pre)
                h2 = module.register_forward_hook(post)
                self._hooks.extend([h1, h2])

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_summary(self) -> dict[str, dict]:
        """Aggregate timings by category (Attention, FFN, etc.)."""
        categories = defaultdict(list)

        for name, times in self.timings.items():
            # Strip the index suffix to group by category
            base = name.rsplit("_", 1)[0] if "_" in name and name.rsplit("_", 1)[1].isdigit() else name
            categories[base].extend(times)

        summary = {}
        for cat, times in categories.items():
            if times:
                total = sum(times)
                summary[cat] = {
                    "total_ms": total,
                    "calls": len(times),
                    "avg_ms": total / len(times),
                }

        return summary


def profile_pipeline(
    model_name: Optional[str] = None,
    prompt: str = "The transformer architecture revolutionized natural language processing by introducing",
    max_new_tokens: int = 32,
    num_runs: int = 5,
):
    """
    Profile each stage of the inference pipeline and report
    where time is being spent.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model

    console.print(f"\n[bold]Pipeline Profiler: {model_name}[/]\n")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    # Measure tokenization time separately
    tokenize_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokenize_times.append((time.perf_counter() - start) * 1000)

    # Attach layer-level profiling
    timer = LayerTimer()
    timer.attach(model)

    # Run inference
    for _ in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    timer.detach()

    # Measure sampling/decode separately
    decode_times = []
    for _ in range(num_runs):
        dummy_ids = list(range(100))
        start = time.perf_counter()
        tokenizer.decode(dummy_ids, skip_special_tokens=True)
        decode_times.append((time.perf_counter() - start) * 1000)

    # Build report
    summary = timer.get_summary()
    total_model_time = sum(v["total_ms"] for v in summary.values())

    table = Table(title="Pipeline Stage Breakdown", show_header=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Total (ms)", justify="right", style="green")
    table.add_column("% of Model", justify="right", style="yellow")
    table.add_column("Calls", justify="right")
    table.add_column("Avg (ms)", justify="right")

    # Add tokenization
    avg_tok = sum(tokenize_times) / len(tokenize_times)
    table.add_row("Tokenization", f"{avg_tok:.2f}", "—", str(num_runs), f"{avg_tok:.2f}")

    # Model stages sorted by time
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]["total_ms"]):
        pct = (stats["total_ms"] / total_model_time * 100) if total_model_time > 0 else 0
        table.add_row(
            stage,
            f"{stats['total_ms']:.1f}",
            f"{pct:.1f}%",
            str(stats["calls"]),
            f"{stats['avg_ms']:.3f}",
        )

    # Add decoding
    avg_dec = sum(decode_times) / len(decode_times)
    table.add_row("Detokenization", f"{avg_dec:.2f}", "—", str(num_runs), f"{avg_dec:.2f}")

    console.print(table)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "tokenization_ms": avg_tok,
        "model_stages": summary,
        "detokenization_ms": avg_dec,
    }


if __name__ == "__main__":
    profile_pipeline(num_runs=3)
