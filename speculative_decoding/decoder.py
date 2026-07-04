# TokenForge GPU-Accelerated LLM Inference Platform
"""
Speculative decoding loop.

Orchestrates the draft-verify cycle:
1. Draft model generates K candidate tokens (fast)
2. Target model verifies all K in a single pass
3. Accept matching tokens, reject at first mismatch
4. Repeat until max_new_tokens reached

The speedup comes from replacing K autoregressive steps of the
target model with 1 draft generation + 1 verification pass.
"""

import time
from typing import Optional

import torch
from rich.console import Console

from core.config import get_config
from speculative_decoding.draft_model import DraftModel
from speculative_decoding.verifier import TargetVerifier

console = Console()


class SpeculativeDecoder:
    """
    Combines draft and target models for speculative decoding.
    """

    def __init__(
        self,
        draft_model_name: Optional[str] = None,
        target_model_name: Optional[str] = None,
        num_speculative_tokens: int = 5,
    ):
        cfg = get_config()

        self.num_speculative = num_speculative_tokens
        self.draft = DraftModel(model_name=draft_model_name)
        self.target = TargetVerifier(model_name=target_model_name)

        # They should ideally share a tokenizer
        self.tokenizer = self.target.tokenizer

        self._total_draft_tokens = 0
        self._total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        if self._total_draft_tokens == 0:
            return 0.0
        return self._total_accepted / self._total_draft_tokens

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> tuple[str, dict]:
        """
        Generate text using speculative decoding.

        Returns:
            output_text: The generated text
            stats: Dictionary with performance statistics
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.draft.device)
        prompt_len = input_ids.shape[1]

        current_ids = input_ids.clone()
        generated = 0
        draft_calls = 0
        verify_calls = 0

        start_time = time.perf_counter()
        first_token_time = None

        while generated < max_new_tokens:
            # How many tokens to speculate
            remaining = max_new_tokens - generated
            K = min(self.num_speculative, remaining)

            # Draft
            candidate_tokens, draft_probs = self.draft.generate_candidates(
                current_ids, num_candidates=K, temperature=temperature
            )
            draft_calls += 1

            # Verify
            accepted_tokens, num_accepted = self.target.verify(
                current_ids, candidate_tokens, draft_probs,
                temperature=temperature,
            )
            verify_calls += 1

            self._total_draft_tokens += K
            self._total_accepted += min(num_accepted, K)

            # Append accepted tokens
            current_ids = torch.cat([current_ids, accepted_tokens], dim=1)
            generated += accepted_tokens.shape[1]

            if first_token_time is None:
                first_token_time = time.perf_counter()

        total_time = time.perf_counter() - start_time

        output_text = self.tokenizer.decode(
            current_ids[0][prompt_len:], skip_special_tokens=True
        )

        stats = {
            "total_time_s": total_time,
            "tokens_generated": generated,
            "tokens_per_sec": generated / max(total_time, 1e-9),
            "draft_calls": draft_calls,
            "verify_calls": verify_calls,
            "acceptance_rate": self.acceptance_rate,
            "ttft_s": (first_token_time - start_time) if first_token_time else None,
            "speculative_tokens": self.num_speculative,
        }

        return output_text, stats

    def cleanup(self):
        self.draft.cleanup()
        self.target.cleanup()
