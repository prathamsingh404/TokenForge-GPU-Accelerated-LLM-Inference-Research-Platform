"""
Prefix caching benchmark.

Simulates a workload where many requests share common prefixes
(like a system prompt) and measures the compute savings from
caching vs full re-computation.
"""

import time
from typing import Optional

from rich.console import Console
from rich.table import Table

from core.config import get_config
from prefix_caching.cache_engine import PrefixCacheEngine

console = Console()

# Simulated workload: shared system prompt + varied user queries
SYSTEM_PREFIX = (
    "You are a helpful AI assistant specializing in physics and mathematics. "
    "Provide clear, accurate explanations suitable for university students. "
    "Use examples where helpful. "
)

USER_QUERIES = [
    "Explain the Schrödinger equation.",
    "What is the eigenvalue problem?",
    "Describe Fourier transforms.",
    "How does quantum tunneling work?",
    "What is the Heisenberg uncertainty principle?",
    "Explain Lagrangian mechanics.",
    "What are Maxwell's equations?",
    "Describe the photoelectric effect.",
    "What is special relativity?",
    "How does a Bose-Einstein condensate form?",
]


def run_prefix_caching_benchmark(
    model_name: Optional[str] = None,
    max_new_tokens: int = 32,
    num_queries: int = 8,
):
    """
    Run queries with and without prefix caching and compare.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.small_model
    queries = USER_QUERIES[:num_queries]

    console.print(f"\n[bold magenta]{'='*60}[/]")
    console.print(f"[bold magenta]Prefix Caching Benchmark[/]")
    console.print(f"[bold magenta]Model: {model_name}[/]")
    console.print(f"[bold magenta]Queries: {num_queries} with shared prefix[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # --- Without prefix caching ---
    console.print("[cyan]→ Without prefix caching (full prefill each time)...[/]")
    engine_no_cache = PrefixCacheEngine(
        model_name=model_name, max_cache_entries=0
    )

    no_cache_times = []
    for i, query in enumerate(queries):
        prompt = SYSTEM_PREFIX + query
        _, stats = engine_no_cache.generate(prompt, max_new_tokens=max_new_tokens)
        no_cache_times.append(stats["total_time_s"])
        console.print(
            f"  Query {i+1}: {stats['total_time_s']:.3f}s "
            f"({stats['tokens_per_sec']:.1f} tok/s)"
        )

    engine_no_cache.cleanup()

    # --- With prefix caching ---
    console.print("\n[cyan]→ With prefix caching...[/]")
    engine_cached = PrefixCacheEngine(
        model_name=model_name, max_cache_entries=50
    )

    cached_times = []
    for i, query in enumerate(queries):
        prompt = SYSTEM_PREFIX + query
        _, stats = engine_cached.generate(prompt, max_new_tokens=max_new_tokens)
        cached_times.append(stats["total_time_s"])
        hit = "HIT" if stats["cache_hit"] else "MISS"
        console.print(
            f"  Query {i+1}: {stats['total_time_s']:.3f}s "
            f"({stats['tokens_per_sec']:.1f} tok/s) [{hit}]"
        )

    cache_stats = engine_cached.get_cache_stats()
    engine_cached.cleanup()

    # Results
    avg_no_cache = sum(no_cache_times) / len(no_cache_times)
    avg_cached = sum(cached_times) / len(cached_times)
    speedup = avg_no_cache / max(avg_cached, 1e-9)

    # First query is always a miss, so compare subsequent queries
    subsequent_no_cache = sum(no_cache_times[1:]) / max(len(no_cache_times) - 1, 1)
    subsequent_cached = sum(cached_times[1:]) / max(len(cached_times) - 1, 1)
    subsequent_speedup = subsequent_no_cache / max(subsequent_cached, 1e-9)

    table = Table(title="\nPrefix Caching Results")
    table.add_column("Metric", style="cyan")
    table.add_column("No Cache", justify="right", style="green")
    table.add_column("Cached", justify="right", style="yellow")

    table.add_row("Avg Time (all)", f"{avg_no_cache:.3f}s", f"{avg_cached:.3f}s")
    table.add_row("Avg Time (cached)", f"{subsequent_no_cache:.3f}s", f"{subsequent_cached:.3f}s")
    table.add_row("Speedup (cached)", "1.00x", f"{subsequent_speedup:.2f}x")
    table.add_row("Total Time", f"{sum(no_cache_times):.2f}s", f"{sum(cached_times):.2f}s")
    table.add_row("Cache Hit Rate", "—", f"{cache_stats['hit_rate']:.1%}")
    table.add_row("Cache Entries", "—", str(cache_stats["entries"]))

    console.print(table)

    return {
        "avg_no_cache": avg_no_cache,
        "avg_cached": avg_cached,
        "speedup": speedup,
        "cache_stats": cache_stats,
    }


if __name__ == "__main__":
    run_prefix_caching_benchmark(num_queries=6, max_new_tokens=24)
