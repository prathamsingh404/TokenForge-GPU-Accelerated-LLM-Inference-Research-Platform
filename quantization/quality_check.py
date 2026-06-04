"""
Output quality checker for quantized models.

Compares the generation quality between precision levels by measuring
perplexity on a reference dataset and checking for output divergence.
This isn't a full accuracy evaluation, but a quick sanity check to
verify quantization didn't break the model.
"""

import torch
import math
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rich.console import Console
from rich.table import Table

from core.config import get_config

console = Console()

# Reference texts for perplexity measurement
EVAL_TEXTS = [
    "The Higgs boson is an elementary particle in the Standard Model of particle physics. It was first hypothesized in 1964 and experimentally confirmed in 2012 at CERN's Large Hadron Collider.",
    "Gradient descent is an iterative optimization algorithm used to find the minimum of a function. In machine learning, it adjusts model parameters to minimize the loss function by computing partial derivatives.",
    "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes. The Human Genome Project completed its mapping in 2003 after 13 years of research.",
    "Transformer architectures use self-attention mechanisms to process sequences in parallel, unlike recurrent models that process tokens sequentially. This parallelism enables significantly faster training on modern hardware.",
]


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of a text under the given model."""
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())


def check_output_similarity(
    model_a, model_b,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> dict:
    """Compare outputs of two models on the same prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model_a.device)

    with torch.no_grad():
        out_a = model_a.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out_b = model_b.generate(
            **tokenizer(prompt, return_tensors="pt").to(model_b.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
    text_b = tokenizer.decode(out_b[0], skip_special_tokens=True)

    # Simple token-level overlap
    tokens_a = set(tokenizer.encode(text_a))
    tokens_b = set(tokenizer.encode(text_b))
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)

    return {
        "output_a": text_a,
        "output_b": text_b,
        "token_overlap": overlap,
    }


def run_quality_check(model_name: Optional[str] = None):
    """
    Run perplexity comparisons across FP16, INT8, INT4 precision levels.
    """
    cfg = get_config()
    model_name = model_name or cfg.models.medium_model

    console.print(f"\n[bold]Quality Check: {model_name}[/]\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # FP16
    console.print("[dim]Loading FP16...[/]")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model_fp16.eval()

    ppls = [compute_perplexity(model_fp16, tokenizer, t) for t in EVAL_TEXTS]
    results["fp16"] = sum(ppls) / len(ppls)
    del model_fp16
    torch.cuda.empty_cache()

    # INT8
    console.print("[dim]Loading INT8...[/]")
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda",
    )
    model_int8.eval()

    ppls = [compute_perplexity(model_int8, tokenizer, t) for t in EVAL_TEXTS]
    results["int8"] = sum(ppls) / len(ppls)
    del model_int8
    torch.cuda.empty_cache()

    # INT4
    console.print("[dim]Loading INT4...[/]")
    model_int4 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map="cuda",
    )
    model_int4.eval()

    ppls = [compute_perplexity(model_int4, tokenizer, t) for t in EVAL_TEXTS]
    results["int4"] = sum(ppls) / len(ppls)
    del model_int4
    torch.cuda.empty_cache()

    # Display
    table = Table(title="Perplexity Comparison (lower is better)")
    table.add_column("Precision", style="cyan")
    table.add_column("Avg Perplexity", justify="right", style="green")
    table.add_column("Δ vs FP16", justify="right", style="yellow")

    baseline = results["fp16"]
    for prec, ppl in results.items():
        delta = ((ppl - baseline) / baseline * 100) if baseline > 0 else 0
        table.add_row(
            prec.upper(),
            f"{ppl:.2f}",
            f"{delta:+.1f}%" if prec != "fp16" else "—",
        )

    console.print(table)
    return results


if __name__ == "__main__":
    run_quality_check()
