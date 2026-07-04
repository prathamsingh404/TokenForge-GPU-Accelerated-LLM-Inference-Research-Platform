# TokenForge GPU-Accelerated LLM Inference Platform
"""
Target model verifier for speculative decoding.

The verifier runs the target (large) model on the draft tokens
in a single forward pass, then uses rejection sampling to decide
which draft tokens to accept. Accepted tokens are kept; the first
rejected token is replaced with the target model's choice.
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import get_config


class TargetVerifier:
    """
    Loads the target model and verifies draft tokens.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        cfg = get_config()
        self.model_name = model_name or cfg.models.target_model

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
    def verify(
        self,
        input_ids: torch.Tensor,
        candidate_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """
        Verify draft tokens using rejection sampling.

        The key insight: we can evaluate ALL candidate positions in
        a single forward pass of the target model, then decide
        acceptance token by token.

        Args:
            input_ids: Original prompt tokens (1, prompt_len)
            candidate_tokens: Draft tokens to verify (1, K)
            draft_probs: Draft model's distributions (K, vocab_size)
            temperature: Sampling temperature

        Returns:
            accepted_tokens: Tokens that pass verification
            num_accepted: How many draft tokens were accepted
        """
        K = candidate_tokens.shape[1]

        # Concatenate prompt + draft tokens for single forward pass
        full_sequence = torch.cat([input_ids, candidate_tokens], dim=1)

        outputs = self.model(full_sequence)
        target_logits = outputs.logits

        # Extract logits at draft positions
        # Position i in the prompt predicts position i+1
        prompt_len = input_ids.shape[1]
        relevant_logits = target_logits[:, prompt_len - 1 : prompt_len + K - 1, :]

        if temperature > 0:
            target_probs = torch.softmax(relevant_logits.squeeze(0) / temperature, dim=-1)
        else:
            target_probs = torch.softmax(relevant_logits.squeeze(0), dim=-1)

        # Rejection sampling: accept token i if
        # target_prob[token_i] / draft_prob[token_i] >= uniform(0,1)
        accepted = []
        num_accepted = 0

        for i in range(K):
            token = candidate_tokens[0, i].item()
            p_target = target_probs[i, token].item()
            p_draft = draft_probs[i, token].item()

            # Acceptance probability
            if p_draft > 0:
                accept_ratio = min(1.0, p_target / p_draft)
            else:
                accept_ratio = 1.0 if p_target > 0 else 0.0

            r = torch.rand(1).item()
            if r < accept_ratio:
                accepted.append(token)
                num_accepted += 1
            else:
                # Reject: sample from adjusted distribution
                # p_adjusted = max(0, p_target - p_draft) / Z
                adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                adjusted_sum = adjusted.sum()
                if adjusted_sum > 0:
                    adjusted = adjusted / adjusted_sum
                    corrected_token = torch.multinomial(adjusted, 1).item()
                else:
                    corrected_token = torch.argmax(target_probs[i]).item()

                accepted.append(corrected_token)
                num_accepted += 1
                break  # Stop at first rejection

        if num_accepted == K:
            # All accepted — bonus: sample one more from target
            final_logits = target_logits[:, -1, :]
            if temperature > 0:
                final_probs = torch.softmax(final_logits / temperature, dim=-1)
            else:
                final_probs = torch.softmax(final_logits, dim=-1)
            bonus_token = torch.argmax(final_probs, dim=-1).item()
            accepted.append(bonus_token)

        accepted_tensor = torch.tensor([accepted], device=self.device)
        return accepted_tensor, num_accepted

    def cleanup(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
