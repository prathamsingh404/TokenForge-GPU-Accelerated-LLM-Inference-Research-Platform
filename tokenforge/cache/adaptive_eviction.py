"""
Importance-based adaptive KV cache eviction.

Instead of naive LRU or FIFO eviction, this policy uses attention
score history to determine which cached tokens are actually
important for future predictions.

Research basis:
- H2O (Heavy-Hitter Oracle): Keep tokens with highest cumulative attention
- Scissorhands: Identify "pivotal" tokens via attention patterns
- StreamingLLM: Keep initial tokens + recent window (attention sinks)

This implementation combines attention accumulation with a
configurable eviction policy.
"""

import torch
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class EvictionPolicy(Enum):
    LRU = "lru"
    ATTENTION_SCORE = "attention_score"
    HEAVY_HITTER = "heavy_hitter"
    STREAMING = "streaming"


@dataclass
class EvictionStats:
    """Statistics about cache eviction behavior."""
    total_evictions: int = 0
    total_entries: int = 0
    entries_retained: int = 0
    avg_importance_evicted: float = 0.0
    avg_importance_retained: float = 0.0


class AdaptiveEvictionCache:
    """
    KV cache with importance-based eviction.

    Tracks per-token importance scores and evicts the least important
    tokens when memory pressure requires it, instead of simply
    removing the oldest tokens.

    Supports multiple eviction policies:
    - attention_score: Evict tokens with lowest cumulative attention
    - heavy_hitter: Keep the top-K most attended tokens (H2O paper)
    - streaming: Keep first N tokens + last M tokens (StreamingLLM)
    - lru: Fallback to least recently used
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        budget: int = 2048,
        policy: EvictionPolicy = EvictionPolicy.HEAVY_HITTER,
        sink_tokens: int = 4,
        recent_window: int = 256,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.budget = budget
        self.policy = policy
        self.sink_tokens = sink_tokens
        self.recent_window = recent_window
        self.device = device

        # Importance scores: cumulative attention received by each position
        self._importance: Optional[torch.Tensor] = None  # [layers, seq_len]
        self._cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._current_len = 0
        self._stats = EvictionStats()

    def allocate(self):
        """Pre-allocate cache and importance tracking."""
        shape = (1, self.num_heads, self.max_seq_len, self.head_dim)

        self._cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(shape, dtype=torch.float16, device=self.device)
            v = torch.zeros(shape, dtype=torch.float16, device=self.device)
            self._cache.append((k, v))

        self._importance = torch.zeros(
            self.num_layers, self.max_seq_len,
            device=self.device,
        )
        self._current_len = 0

    def update_importance(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
    ):
        """
        Update importance scores from attention weights.

        Called after each attention computation. Accumulates attention
        received by each position across all heads and steps.

        Args:
            layer_idx: Which transformer layer.
            attention_weights: [batch, heads, query_len, kv_len]
        """
        if self._importance is None:
            return

        # Sum attention received across heads and query positions
        # Shape: [kv_len]
        attn_received = attention_weights.sum(dim=(0, 1, 2))
        kv_len = min(attn_received.shape[0], self._current_len)
        self._importance[layer_idx, :kv_len] += attn_received[:kv_len]

    def maybe_evict(self, layer_idx: int) -> bool:
        """
        Evict entries if cache exceeds budget.

        Returns True if eviction occurred.
        """
        if self._current_len <= self.budget:
            return False

        if self.policy == EvictionPolicy.HEAVY_HITTER:
            self._evict_heavy_hitter(layer_idx)
        elif self.policy == EvictionPolicy.STREAMING:
            self._evict_streaming(layer_idx)
        elif self.policy == EvictionPolicy.ATTENTION_SCORE:
            self._evict_by_attention(layer_idx)
        else:
            self._evict_lru(layer_idx)

        return True

    def _evict_heavy_hitter(self, layer_idx: int):
        """Keep the top-budget tokens by importance (H2O strategy)."""
        scores = self._importance[layer_idx, :self._current_len]

        # Always keep sink tokens (first N) and recent window (last M)
        keep_mask = torch.zeros(self._current_len, dtype=torch.bool, device=self.device)
        keep_mask[:self.sink_tokens] = True
        keep_mask[-self.recent_window:] = True

        # For the middle section, keep highest-importance tokens
        middle_start = self.sink_tokens
        middle_end = self._current_len - self.recent_window
        if middle_end > middle_start:
            middle_budget = self.budget - self.sink_tokens - self.recent_window
            if middle_budget > 0:
                middle_scores = scores[middle_start:middle_end]
                _, top_indices = middle_scores.topk(
                    min(middle_budget, middle_end - middle_start),
                )
                keep_mask[middle_start + top_indices] = True

        self._apply_eviction_mask(layer_idx, keep_mask)

    def _evict_streaming(self, layer_idx: int):
        """StreamingLLM: keep first N (sinks) + last M (recent)."""
        keep_mask = torch.zeros(self._current_len, dtype=torch.bool, device=self.device)
        keep_mask[:self.sink_tokens] = True
        keep_mask[-self.recent_window:] = True
        self._apply_eviction_mask(layer_idx, keep_mask)

    def _evict_by_attention(self, layer_idx: int):
        """Evict tokens with lowest cumulative attention score."""
        scores = self._importance[layer_idx, :self._current_len]
        _, keep_indices = scores.topk(self.budget)
        keep_indices, _ = keep_indices.sort()

        keep_mask = torch.zeros(self._current_len, dtype=torch.bool, device=self.device)
        keep_mask[keep_indices] = True
        self._apply_eviction_mask(layer_idx, keep_mask)

    def _evict_lru(self, layer_idx: int):
        """Simple: keep the most recent tokens."""
        keep_mask = torch.zeros(self._current_len, dtype=torch.bool, device=self.device)
        keep_mask[-self.budget:] = True
        self._apply_eviction_mask(layer_idx, keep_mask)

    def _apply_eviction_mask(self, layer_idx: int, keep_mask: torch.Tensor):
        """Apply a boolean mask to compact the cache."""
        if self._cache is None:
            return

        k, v = self._cache[layer_idx]
        kept_k = k[:, :, keep_mask, :]
        kept_v = v[:, :, keep_mask, :]

        new_len = kept_k.shape[2]

        # Track stats
        evicted_count = self._current_len - new_len
        self._stats.total_evictions += evicted_count
        self._stats.entries_retained = new_len

        # Compact into cache
        k.zero_()
        v.zero_()
        k[:, :, :new_len, :] = kept_k
        v[:, :, :new_len, :] = kept_v

        # Compact importance scores
        kept_importance = self._importance[layer_idx, keep_mask]
        self._importance[layer_idx].zero_()
        self._importance[layer_idx, :new_len] = kept_importance

        self._current_len = new_len

    def get_cache(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get the current cache state for a layer."""
        if self._cache is None or self._current_len == 0:
            return None
        k, v = self._cache[layer_idx]
        return k[:, :, :self._current_len, :], v[:, :, :self._current_len, :]

    def get_stats(self) -> EvictionStats:
        self._stats.total_entries = self._current_len * self.num_layers
        return self._stats

    def free(self):
        self._cache = None
        self._importance = None
        self._current_len = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        self.allocate()
        return self

    def __exit__(self, *exc):
        self.free()
