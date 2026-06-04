"""
KV cache manager for transformer inference.

Implements pre-allocated key-value cache buffers and tracks memory
growth during generation. The KV cache stores computed attention
key and value tensors so they don't need to be recomputed for
previously seen tokens — the single biggest optimization in
autoregressive decoding.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheStats:
    """Memory and utilization stats for the KV cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    current_seq_len: int
    dtype: str
    bytes_per_entry: int

    @property
    def total_entries(self) -> int:
        # 2 for K and V, across all layers and heads
        return 2 * self.num_layers * self.num_heads * self.max_seq_len * self.head_dim

    @property
    def total_bytes(self) -> int:
        return self.total_entries * self.bytes_per_entry

    @property
    def used_bytes(self) -> int:
        ratio = self.current_seq_len / max(self.max_seq_len, 1)
        return int(self.total_bytes * ratio)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def used_mb(self) -> float:
        return self.used_bytes / (1024 ** 2)


class KVCacheManager:
    """
    Manages pre-allocated KV cache buffers.

    This mirrors what serving systems like vLLM do internally —
    pre-allocate GPU memory for the cache to avoid allocation
    overhead during generation.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.device = device

        self._cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._current_len = 0

    def allocate(self):
        """Pre-allocate cache buffers on GPU."""
        cache_shape = (
            self.max_batch_size,
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )

        self._cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            v = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
            self._cache.append((k, v))

        self._current_len = 0

    def free(self):
        """Release cache memory."""
        if self._cache is not None:
            del self._cache
            self._cache = None
            self._current_len = 0
            torch.cuda.empty_cache()

    def get_cache(self) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Get the current cache state for passing to the model."""
        if self._cache is None or self._current_len == 0:
            return None

        # Return sliced view up to current length
        return [
            (k[:, :, :self._current_len, :], v[:, :, :self._current_len, :])
            for k, v in self._cache
        ]

    def update(self, new_cache: list[tuple[torch.Tensor, torch.Tensor]]):
        """Update cache with new KV pairs from the model output."""
        if self._cache is None:
            raise RuntimeError("Cache not allocated. Call allocate() first.")

        for layer_idx, (new_k, new_v) in enumerate(new_cache):
            seq_len = new_k.shape[2]
            self._cache[layer_idx][0][:, :, :seq_len, :] = new_k
            self._cache[layer_idx][1][:, :, :seq_len, :] = new_v

        self._current_len = new_cache[0][0].shape[2]

    def get_stats(self) -> CacheStats:
        bytes_per = torch.tensor([], dtype=self.dtype).element_size()
        return CacheStats(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            current_seq_len=self._current_len,
            dtype=str(self.dtype),
            bytes_per_entry=bytes_per,
        )

    def __enter__(self):
        self.allocate()
        return self

    def __exit__(self, *exc):
        self.free()


def estimate_kv_cache_size(
    model_name_or_config,
    max_seq_len: int = 2048,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Estimate KV cache memory for a given model configuration
    without actually loading the model.
    """
    from transformers import AutoConfig

    if isinstance(model_name_or_config, str):
        config = AutoConfig.from_pretrained(model_name_or_config)
    else:
        config = model_name_or_config

    num_layers = config.num_hidden_layers
    num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    bytes_per = torch.tensor([], dtype=dtype).element_size()

    # 2 tensors (K, V) × layers × batch × heads × seq × dim × bytes
    total_bytes = 2 * num_layers * batch_size * num_heads * max_seq_len * head_dim * bytes_per
    per_token_bytes = 2 * num_layers * num_heads * head_dim * bytes_per

    return {
        "total_mb": total_bytes / (1024 ** 2),
        "per_token_bytes": per_token_bytes,
        "per_token_kb": per_token_bytes / 1024,
        "num_layers": num_layers,
        "num_kv_heads": num_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
    }
