"""
Static batching for LLM inference.

Groups multiple requests into a fixed-size batch, pads to uniform
length, and processes them in a single forward pass. This is the
simplest batching strategy — baseline for continuous batching comparisons.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import get_config
from core.metrics import TimingResult


SAMPLE_PROMPTS = [
    "What is the theory of relativity?",
    "How do neural networks learn?",
    "Explain quantum entanglement in simple terms.",
    "What causes the northern lights?",
    "Describe the process of photosynthesis.",
    "How does a compiler translate source code?",
    "What are the properties of superconductors?",
    "Explain the difference between TCP and UDP.",
    "How does CRISPR gene editing work?",
    "What is the halting problem in computer science?",
    "Describe the architecture of a modern CPU.",
    "How do black holes form?",
    "What is the significance of Euler's identity?",
    "Explain how transformers handle long-range dependencies.",
    "What are the fundamentals of information theory?",
    "How does public key cryptography work?",
    "What is the double-slit experiment?",
    "Explain the concept of entropy in thermodynamics.",
    "How do GPUs achieve parallel computation?",
    "What is the P vs NP problem?",
    "Describe the mechanism of nuclear fusion.",
    "How does gradient descent converge?",
    "What are the principles behind MRI imaging?",
    "Explain the Byzantine Generals Problem.",
    "What is the role of attention in sequence models?",
    "How do vaccines train the immune system?",
    "Describe the structure of the internet.",
    "What is Gödel's incompleteness theorem?",
    "How does reinforcement learning differ from supervised?",
    "What causes earthquakes?",
    "Explain the CAP theorem.",
    "How do lasers produce coherent light?",
]


class StaticBatcher:
    """
    Batches multiple prompts together with padding for parallel inference.

    Usage:
        batcher = StaticBatcher(model, tokenizer, batch_size=8)
        results = batcher.process(prompts, max_new_tokens=64)
    """

    def __init__(self, model, tokenizer, batch_size: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def _prepare_batch(self, prompts: list[str]):
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

    def process(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
    ) -> tuple[list[str], float]:
        """
        Process prompts in fixed-size batches. Returns (outputs, total_time).
        """
        all_outputs = []
        total_time = 0.0

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            inputs = self._prepare_batch(batch_prompts)
            input_len = inputs["input_ids"].shape[1]

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start
            total_time += batch_time

            for j in range(output_ids.shape[0]):
                text = self.tokenizer.decode(
                    output_ids[j][input_len:],
                    skip_special_tokens=True,
                )
                all_outputs.append(text)

        return all_outputs, total_time

    def benchmark_single_batch(
        self,
        prompts: Optional[list[str]] = None,
        max_new_tokens: int = 64,
    ) -> TimingResult:
        """Run a single batch and return timing."""
        if prompts is None:
            prompts = SAMPLE_PROMPTS[:self.batch_size]
        else:
            prompts = prompts[:self.batch_size]

        # Pad if we don't have enough prompts
        while len(prompts) < self.batch_size:
            prompts.append(prompts[0])

        inputs = self._prepare_batch(prompts)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_new_tokens = sum(
            (output_ids[i] != self.tokenizer.pad_token_id).sum().item() - input_len
            for i in range(output_ids.shape[0])
        )

        return TimingResult(
            total_time_s=elapsed,
            tokens_generated=max(total_new_tokens, 1),
            input_tokens=input_len * self.batch_size,
        )
