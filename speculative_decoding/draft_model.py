"""
Draft model handler for speculative decoding.

The draft model is a small, fast model that generates candidate
tokens cheaply. These candidates are then verified by the larger
target model in a single forward pass.

Good draft models are:
- Same tokenizer as the target (critical)
- Much smaller / faster
- Reasonably aligned with the target's distribution
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import get_config


class DraftModel:
    """
    Wraps a small language model for fast candidate generation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        cfg = get_config()
        self.model_name = model_name or cfg.models.draft_model

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="cuda",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        num_candidates: int = 5,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively generate num_candidates tokens.

        Returns:
            candidate_tokens: shape (1, num_candidates) — the draft tokens
            draft_probs: shape (num_candidates, vocab_size) — probability
                         distributions at each draft position
        """
        current = input_ids.clone()
        candidate_tokens = []
        draft_probs = []

        for _ in range(num_candidates):
            outputs = self.model(current)
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            candidate_tokens.append(next_token)
            draft_probs.append(probs.squeeze(0))

            current = torch.cat([current, next_token], dim=1)

        candidate_tokens = torch.cat(candidate_tokens, dim=1)
        draft_probs = torch.stack(draft_probs, dim=0)

        return candidate_tokens, draft_probs

    def cleanup(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
