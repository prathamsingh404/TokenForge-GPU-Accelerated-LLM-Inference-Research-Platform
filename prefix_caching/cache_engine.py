"""
Prefix-aware inference engine.

Wraps a transformer model with the prefix trie cache, automatically
detecting and reusing shared prefixes across requests.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

from core.config import get_config
from prefix_caching.trie_cache import PrefixTrieCache

console = Console()


class PrefixCacheEngine:
    """
    Inference engine with prefix KV cache reuse.

    When a new request arrives, the engine:
    1. Tokenizes the prompt
    2. Searches the trie for the longest matching cached prefix
    3. If found, reuses the cached KV states and only computes
       the remaining tokens (partial prefill)
    4. If not found, runs full prefill and caches the result
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_cache_entries: int = 50,
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

        self.cache = PrefixTrieCache(max_entries=max_cache_entries)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> tuple[str, dict]:
        """
        Generate text with prefix cache awareness.

        Returns (output_text, stats_dict).
        """
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)

        start_time = time.perf_counter()
        cache_hit = False
        prefix_len = 0

        # Check for cached prefix
        match = self.cache.longest_prefix_match(input_ids)

        if match is not None:
            cache_hit = True
            prefix_len = match.prefix_length
            past_kv = match.kv_cache

            # Only process the uncached suffix
            suffix_ids = input_tensor[:, prefix_len:]

            if suffix_ids.shape[1] > 0:
                outputs = self.model(
                    suffix_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = outputs.past_key_values
            # else: entire prompt was cached
        else:
            # Full prefill
            outputs = self.model(input_tensor, use_cache=True)
            past_kv = outputs.past_key_values

            # Cache this prefix for future reuse
            self.cache.insert(input_ids, past_kv)

        prefill_time = time.perf_counter() - start_time

        # Decode loop
        generated_ids = list(input_ids)
        for _ in range(max_new_tokens):
            last_token = torch.tensor([[generated_ids[-1]]], device=self.device)
            outputs = self.model(
                last_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        total_time = time.perf_counter() - start_time
        new_tokens = len(generated_ids) - len(input_ids)

        output_text = self.tokenizer.decode(
            generated_ids[len(input_ids):],
            skip_special_tokens=True,
        )

        stats = {
            "cache_hit": cache_hit,
            "prefix_reused_tokens": prefix_len if cache_hit else 0,
            "prefill_time_s": prefill_time,
            "total_time_s": total_time,
            "tokens_generated": new_tokens,
            "tokens_per_sec": new_tokens / max(total_time, 1e-9),
        }

        return output_text, stats

    def get_cache_stats(self) -> dict:
        return self.cache.get_stats()

    def cleanup(self):
        self.cache.clear()
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
