"""
Continuous batching engine.

Implements iteration-level scheduling where the GPU processes
a dynamic batch that changes composition every decode step.
Requests enter and leave independently, keeping utilization high.

This is a simplified version of the engine powering vLLM, Orca,
and similar production serving systems.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

from core.config import get_config
from continuous_batching.scheduler import ContinuousBatchScheduler, ScheduledBatch
from continuous_batching.request_queue import InferenceRequest, RequestStatus

console = Console()


class ContinuousBatchingEngine:
    """
    Core engine that runs the continuous batching loop.

    The main loop:
    1. Scheduler builds a batch from active + new requests
    2. Engine runs one forward pass for the batch
    3. Each request gets its next token
    4. Finished requests are evicted, new ones admitted
    5. Repeat until all requests are served
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_batch_size: int = 8,
        dtype: torch.dtype = torch.float16,
    ):
        cfg = get_config()
        model_name = model_name or cfg.models.small_model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="cuda",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size)

        # Per-request state: maps request_id -> current token sequence
        self._sequences: dict[str, torch.Tensor] = {}
        self._past_kvs: dict[str, Optional[tuple]] = {}

    def submit(self, requests: list[InferenceRequest]):
        """Submit new requests for processing."""
        for req in requests:
            input_ids = self.tokenizer.encode(req.prompt, return_tensors="pt")
            self._sequences[req.request_id] = input_ids.to(self.device)

        self.scheduler.add_requests(requests)

    def step(self) -> int:
        """
        Run one iteration of the continuous batching loop.
        Returns number of tokens generated in this step.
        """
        batch = self.scheduler.schedule()

        if batch.is_empty:
            return 0

        tokens_this_step = 0
        all_requests = batch.prefill_requests + batch.decode_requests

        # Process each request individually for simplicity.
        # A production system would pad and batch these together,
        # but individual processing shows the scheduling concept clearly.
        for req in all_requests:
            if req.request_id not in self._sequences:
                continue

            input_ids = self._sequences[req.request_id]

            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=False)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            self._sequences[req.request_id] = torch.cat(
                [input_ids, next_token], dim=1
            )

            self.scheduler.mark_token_generated(req.request_id)
            tokens_this_step += 1

            # Check completion
            if req.generated_tokens >= req.max_new_tokens:
                seq = self._sequences.pop(req.request_id)
                req.output_text = self.tokenizer.decode(
                    seq[0], skip_special_tokens=True
                )
                self.scheduler.mark_completed(req.request_id)

        return tokens_this_step

    def run_to_completion(self) -> list[InferenceRequest]:
        """Run until all submitted requests are complete."""
        total_tokens = 0
        total_steps = 0
        start_time = time.monotonic()

        while self.scheduler.has_work:
            tokens = self.step()
            total_tokens += tokens
            total_steps += 1

        elapsed = time.monotonic() - start_time

        console.print(
            f"[dim]  Completed in {total_steps} steps, "
            f"{total_tokens} tokens, {elapsed:.2f}s "
            f"({total_tokens/max(elapsed,1e-9):.1f} tok/s)[/]"
        )

        return self.scheduler.get_completed()

    def cleanup(self):
        self._sequences.clear()
        self._past_kvs.clear()
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
