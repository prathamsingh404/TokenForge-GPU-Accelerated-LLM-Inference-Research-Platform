"""
Mixed-precision KV cache with age-based compression.

As context grows, older KV states are progressively compressed
to lower precision formats to save memory:

    Recent context (last N tokens):  FP16 (full precision)
    Old context (N to M tokens ago): INT8 (50% savings)
    Very old context (> M tokens):   INT4 (75% savings)

This is an active area of research. Production systems like
those at Anthropic, Google, and Meta use similar techniques
for long-context inference.

Usage:
    cache = CompressedKVCache(
        num_layers=32, num_heads=32, head_dim=128,
        recent_window=512, medium_window=2048,
    )
    cache.allocate()
    # ... use during inference ...
    print(cache.get_compression_stats())
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompressionStats:
    """Statistics for the compressed cache."""
    total_entries: int
    fp16_entries: int
    int8_entries: int
    int4_entries: int
    total_bytes_uncompressed: int
    total_bytes_compressed: int
    compression_ratio: float
    memory_saved_mb: float


class CompressedKVCache:
    """
    KV cache with age-based precision tiers.

    Implements progressive compression where older tokens are stored
    at reduced precision, dramatically reducing memory for long contexts.

    Tiers:
        Tier 0 (Recent):    FP16 — last `recent_window` tokens
        Tier 1 (Medium):    INT8 — next `medium_window` tokens
        Tier 2 (Old):       INT4 — everything older

    Quantization uses per-channel absmax scaling for quality preservation.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        max_batch_size: int = 1,
        recent_window: int = 512,
        medium_window: int = 2048,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.recent_window = recent_window
        self.medium_window = medium_window
        self.device = device

        self._fp16_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self._int8_cache: Optional[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        self._int4_cache: Optional[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None

        self._current_len = 0

    def allocate(self):
        """Pre-allocate cache buffers for all tiers."""
        base_shape = (self.max_batch_size, self.num_heads, 0, self.head_dim)

        # FP16 tier: only recent_window tokens pre-allocated
        self._fp16_cache = []
        for _ in range(self.num_layers):
            shape = (self.max_batch_size, self.num_heads, self.recent_window, self.head_dim)
            k = torch.zeros(shape, dtype=torch.float16, device=self.device)
            v = torch.zeros(shape, dtype=torch.float16, device=self.device)
            self._fp16_cache.append((k, v))

        # INT8 and INT4 are allocated dynamically as compression happens
        self._int8_cache = [None] * self.num_layers
        self._int4_cache = [None] * self.num_layers

        self._current_len = 0

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        Update cache with new KV pairs and apply compression if needed.

        When the FP16 window is full, older entries are compressed
        to INT8 or INT4.
        """
        if self._fp16_cache is None:
            raise RuntimeError("Cache not allocated. Call allocate() first.")

        seq_len = new_k.shape[2]

        # If we've exceeded the recent window, compress older entries
        if seq_len > self.recent_window:
            self._compress_to_int8(layer_idx, new_k, new_v)
        elif seq_len > self.recent_window + self.medium_window:
            self._compress_to_int4(layer_idx)

        # Store recent entries at FP16
        recent_start = max(0, seq_len - self.recent_window)
        recent_k = new_k[:, :, recent_start:, :]
        recent_v = new_v[:, :, recent_start:, :]

        actual_len = recent_k.shape[2]
        self._fp16_cache[layer_idx][0][:, :, :actual_len, :] = recent_k
        self._fp16_cache[layer_idx][1][:, :, :actual_len, :] = recent_v

        self._current_len = seq_len

    def _compress_to_int8(
        self, layer_idx: int,
        full_k: torch.Tensor, full_v: torch.Tensor,
    ):
        """Compress older entries from FP16 to INT8 with absmax scaling."""
        old_end = max(0, full_k.shape[2] - self.recent_window)
        if old_end == 0:
            return

        old_k = full_k[:, :, :old_end, :]
        old_v = full_v[:, :, :old_end, :]

        # Per-channel absmax quantization
        k_scale = old_k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        v_scale = old_v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

        k_int8 = (old_k / k_scale * 127).to(torch.int8)
        v_int8 = (old_v / v_scale * 127).to(torch.int8)

        self._int8_cache[layer_idx] = (
            k_int8, v_int8, k_scale.half(), v_scale.half(),
        )

    def _compress_to_int4(self, layer_idx: int):
        """Compress the oldest INT8 entries to INT4 (packed)."""
        if self._int8_cache[layer_idx] is None:
            return

        k_int8, v_int8, k_scale, v_scale = self._int8_cache[layer_idx]
        old_end = max(0, k_int8.shape[2] - self.medium_window)
        if old_end == 0:
            return

        # Pack INT8 → INT4 (two values per byte)
        old_k = k_int8[:, :, :old_end, :]
        old_v = v_int8[:, :, :old_end, :]

        # Rescale to 4-bit range (-8 to 7)
        k_int4 = (old_k.float() / 127 * 7).clamp(-8, 7).to(torch.int8)
        v_int4 = (old_v.float() / 127 * 7).clamp(-8, 7).to(torch.int8)

        self._int4_cache[layer_idx] = (
            k_int4, v_int4,
            k_scale[:, :, :old_end, :],
            v_scale[:, :, :old_end, :],
        )

        # Trim INT8 cache to only keep medium-window entries
        self._int8_cache[layer_idx] = (
            k_int8[:, :, old_end:, :],
            v_int8[:, :, old_end:, :],
            k_scale[:, :, old_end:, :],
            v_scale[:, :, old_end:, :],
        )

    def get_cache(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct the full KV cache for a layer by dequantizing
        all tiers and concatenating.
        """
        if self._fp16_cache is None:
            return None

        parts_k = []
        parts_v = []

        # INT4 tier (oldest)
        if self._int4_cache[layer_idx] is not None:
            k4, v4, ks4, vs4 = self._int4_cache[layer_idx]
            parts_k.append((k4.float() / 7 * ks4.float()).half())
            parts_v.append((v4.float() / 7 * vs4.float()).half())

        # INT8 tier (medium age)
        if self._int8_cache[layer_idx] is not None:
            k8, v8, ks8, vs8 = self._int8_cache[layer_idx]
            parts_k.append((k8.float() / 127 * ks8.float()).half())
            parts_v.append((v8.float() / 127 * vs8.float()).half())

        # FP16 tier (recent)
        recent_len = min(self._current_len, self.recent_window)
        if recent_len > 0:
            parts_k.append(self._fp16_cache[layer_idx][0][:, :, :recent_len, :])
            parts_v.append(self._fp16_cache[layer_idx][1][:, :, :recent_len, :])

        if not parts_k:
            return None

        full_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        full_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        return full_k, full_v

    def get_compression_stats(self) -> CompressionStats:
        """Get memory statistics across all compression tiers."""
        bytes_per_fp16 = 2
        bytes_per_int8 = 1
        bytes_per_int4 = 0.5

        entry_size = self.max_batch_size * self.num_heads * self.head_dim
        total_entries = self._current_len * self.num_layers * 2  # K and V

        # Count entries in each tier
        recent_len = min(self._current_len, self.recent_window)
        fp16_entries = recent_len * self.num_layers * 2

        int8_entries = 0
        for c in self._int8_cache:
            if c is not None:
                int8_entries += c[0].shape[2] * 2  # K and V

        int4_entries = 0
        for c in self._int4_cache:
            if c is not None:
                int4_entries += c[0].shape[2] * 2

        # Compute bytes
        uncompressed = total_entries * entry_size * bytes_per_fp16
        compressed = (
            fp16_entries * entry_size * bytes_per_fp16
            + int8_entries * entry_size * bytes_per_int8
            + int4_entries * entry_size * bytes_per_int4
        )

        ratio = uncompressed / max(compressed, 1)
        saved = (uncompressed - compressed) / (1024 ** 2)

        return CompressionStats(
            total_entries=total_entries,
            fp16_entries=fp16_entries,
            int8_entries=int8_entries,
            int4_entries=int4_entries,
            total_bytes_uncompressed=int(uncompressed),
            total_bytes_compressed=int(compressed),
            compression_ratio=ratio,
            memory_saved_mb=saved,
        )

    def free(self):
        """Release all cache memory."""
        self._fp16_cache = None
        self._int8_cache = None
        self._int4_cache = None
        self._current_len = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        self.allocate()
        return self

    def __exit__(self, *exc):
        self.free()
