"""
GPU saturation point finder.

Binary-searches for the batch size where throughput stops scaling
linearly — the "knee" of the throughput curve. Beyond this point,
larger batches give diminishing returns or cause OOM.
"""

import gc
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

from core.config import get_config
from batching.static_batcher import StaticBatcher, SAMPLE_PROMPTS

console = Console()


def find_saturation_point(
    model_name: Optional[str] = None,
    max_batch_size: int = 64,
    max_new_tokens: int = 32,
    num_warmup: int = 2,
    num_measure: int = 3,
    threshold: float = 0.10,
) -> dict:
    """
    Find the batch size where throughput improvement drops below `threshold`
    (default 10%) compared to the previous batch size.

    Returns dict with:
        - saturation_batch_size: the last efficient batch size
        - max_tested: the largest batch size tested
        - throughput_curve: list of (batch_size, tokens_per_sec)
    """
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    console.print(f"\n[bold]Saturation Finder: {model_name}[/]")
    console.print(f"[dim]Threshold: {threshold*100:.0f}% improvement[/]\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    throughput_curve = []
    prev_throughput = 0
    saturation_bs = 1

    bs = 1
    while bs <= max_batch_size:
        try:
            batcher = StaticBatcher(model, tokenizer, batch_size=bs)
            prompts = SAMPLE_PROMPTS[:bs]
            while len(prompts) < bs:
                prompts.append(prompts[0])

            # Warmup
            for _ in range(num_warmup):
                batcher.benchmark_single_batch(prompts=prompts, max_new_tokens=max_new_tokens)

            # Measure
            throughputs = []
            for _ in range(num_measure):
                result = batcher.benchmark_single_batch(prompts=prompts, max_new_tokens=max_new_tokens)
                if result.total_time_s > 0:
                    throughputs.append(result.tokens_generated / result.total_time_s)

            avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
            throughput_curve.append((bs, avg_throughput))

            improvement = (avg_throughput - prev_throughput) / max(prev_throughput, 1e-6)
            console.print(
                f"  BS={bs:>3}: {avg_throughput:>8.1f} tok/s  "
                f"(Δ {improvement*100:+.1f}%)"
            )

            if prev_throughput > 0 and improvement < threshold:
                saturation_bs = bs // 2 if bs > 1 else 1
                console.print(
                    f"\n[yellow]Saturation detected at BS={saturation_bs}[/]"
                )
                break

            prev_throughput = avg_throughput
            saturation_bs = bs
            bs *= 2

        except torch.cuda.OutOfMemoryError:
            console.print(f"  BS={bs}: [red]OOM[/]")
            gc.collect()
            torch.cuda.empty_cache()
            break

    del model, tokenizer
    torch.cuda.empty_cache()

    result = {
        "saturation_batch_size": saturation_bs,
        "max_tested": bs,
        "throughput_curve": throughput_curve,
    }

    console.print(f"\n[bold green]Recommended batch size: {saturation_bs}[/]")
    return result


if __name__ == "__main__":
    find_saturation_point()
