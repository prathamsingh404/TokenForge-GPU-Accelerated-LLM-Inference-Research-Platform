# TokenForge GPU-Accelerated LLM Inference Platform
"""
Tokenizer throughput benchmarks.

Measures encode and decode speed across different tokenizers
and input lengths. Useful for understanding the overhead of
tokenization relative to the GPU inference pipeline.
"""

import time
from typing import Optional

from transformers import AutoTokenizer
from rich.console import Console

from core.config import get_config
from core.metrics import TimingResult
from benchmark_engine.runner import BenchmarkRunner

console = Console()


# Diverse inputs that exercise different vocabulary distributions
BENCHMARK_PROMPTS = [
    "Explain the theory of general relativity and its implications for modern physics.",
    "Write a recursive implementation of merge sort in Python with detailed comments.",
    "The quantum mechanical model of the atom describes electrons as probability distributions rather than fixed orbits.",
    "In distributed systems, the CAP theorem states that a system cannot simultaneously provide consistency, availability, and partition tolerance.",
    "Climate change impacts include rising sea levels, increased frequency of extreme weather events, and disruption of ecosystems worldwide.",
    "SELECT e.name, d.department_name, AVG(s.amount) FROM employees e JOIN departments d ON e.dept_id = d.id JOIN salaries s ON e.id = s.employee_id GROUP BY e.name, d.department_name HAVING AVG(s.amount) > 50000 ORDER BY AVG(s.amount) DESC;",
]


def build_long_input(base_prompts: list[str], target_tokens: int, tokenizer) -> str:
    """Repeat base prompts until we reach approximately target_tokens."""
    text = ""
    while True:
        for prompt in base_prompts:
            text += prompt + " "
            if len(tokenizer.encode(text)) >= target_tokens:
                return text
    return text


def benchmark_tokenizer_encode(
    model_name: str,
    input_lengths: Optional[list[int]] = None,
    num_runs: int = 100,
) -> dict:
    """
    Measure tokenizer encoding speed at various input lengths.

    Returns dict mapping input_length -> tokens_per_second.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if input_lengths is None:
        input_lengths = [16, 64, 256, 1024, 4096]

    results = {}

    for target_len in input_lengths:
        text = build_long_input(BENCHMARK_PROMPTS, target_len, tokenizer)
        actual_len = len(tokenizer.encode(text))

        # Warmup
        for _ in range(10):
            tokenizer.encode(text)

        # Timed
        start = time.perf_counter()
        for _ in range(num_runs):
            tokenizer.encode(text)
        elapsed = time.perf_counter() - start

        tokens_per_sec = (actual_len * num_runs) / elapsed
        results[actual_len] = tokens_per_sec
        console.print(
            f"  Encode {actual_len:>5} tokens: "
            f"{tokens_per_sec:>10,.0f} tok/s "
            f"({elapsed/num_runs*1000:.2f} ms/call)"
        )

    return results


def benchmark_tokenizer_decode(
    model_name: str,
    input_lengths: Optional[list[int]] = None,
    num_runs: int = 100,
) -> dict:
    """Measure tokenizer decoding speed."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if input_lengths is None:
        input_lengths = [16, 64, 256, 1024, 4096]

    results = {}

    for target_len in input_lengths:
        text = build_long_input(BENCHMARK_PROMPTS, target_len, tokenizer)
        token_ids = tokenizer.encode(text)
        actual_len = len(token_ids)

        # Warmup
        for _ in range(10):
            tokenizer.decode(token_ids)

        # Timed
        start = time.perf_counter()
        for _ in range(num_runs):
            tokenizer.decode(token_ids)
        elapsed = time.perf_counter() - start

        tokens_per_sec = (actual_len * num_runs) / elapsed
        results[actual_len] = tokens_per_sec
        console.print(
            f"  Decode {actual_len:>5} tokens: "
            f"{tokens_per_sec:>10,.0f} tok/s "
            f"({elapsed/num_runs*1000:.2f} ms/call)"
        )

    return results


def run_tokenizer_benchmarks(model_name: Optional[str] = None):
    """Full tokenizer benchmark suite."""
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    console.print(f"\n[bold]Tokenizer Benchmark: {model_name}[/]\n")

    console.print("[cyan]Encoding throughput:[/]")
    encode_results = benchmark_tokenizer_encode(model_name)

    console.print(f"\n[cyan]Decoding throughput:[/]")
    decode_results = benchmark_tokenizer_decode(model_name)

    return {
        "model": model_name,
        "encode": encode_results,
        "decode": decode_results,
    }


if __name__ == "__main__":
    run_tokenizer_benchmarks()
