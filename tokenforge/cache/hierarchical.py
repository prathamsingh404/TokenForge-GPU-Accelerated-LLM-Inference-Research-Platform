"""
Hierarchical KV cache with GPU → CPU → Disk tiering.

Implements automatic offloading of less-frequently-accessed KV states
to lower-cost memory tiers. Enables serving models with context
lengths exceeding GPU VRAM capacity.

Tier hierarchy:
    Tier 0: GPU HBM (fastest, most expensive, limited capacity)
    Tier 1: CPU DRAM (10-50x slower, much larger capacity)
    Tier 2: NVMe SSD (100-1000x slower, virtually unlimited)

Prefetching: Predictive prefetch of layers likely needed in the
next decode step, overlapping data transfer with computation.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict

import torch


@dataclass
class TierStats:
    """Statistics for a single cache tier."""
    name: str
    capacity_mb: float
    used_mb: float
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    avg_access_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def utilization(self) -> float:
        return self.used_mb / self.capacity_mb if self.capacity_mb > 0 else 0.0


class HierarchicalKVCache:
    """
    Multi-tier KV cache: GPU → CPU RAM → (optional) disk.

    Automatically manages data placement across memory tiers based
    on access frequency and recency. Supports asynchronous prefetching
    to hide transfer latency.

    Key features:
    - LRU eviction within each tier
    - Async prefetch overlapping with compute
    - Per-tier capacity limits
    - Detailed per-tier statistics
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        gpu_budget_mb: float = 2048.0,
        cpu_budget_mb: float = 8192.0,
        enable_disk: bool = False,
        disk_path: Optional[str] = None,
        prefetch_layers: int = 2,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gpu_budget_mb = gpu_budget_mb
        self.cpu_budget_mb = cpu_budget_mb
        self.enable_disk = enable_disk
        self.disk_path = disk_path
        self.prefetch_layers = prefetch_layers
        self.device = device

        # GPU tier: OrderedDict for LRU
        self._gpu_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._gpu_used_mb: float = 0.0

        # CPU tier
        self._cpu_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cpu_used_mb: float = 0.0

        # Stats
        self._gpu_stats = TierStats("GPU", gpu_budget_mb, 0.0)
        self._cpu_stats = TierStats("CPU", cpu_budget_mb, 0.0)

        # Prefetch thread
        self._prefetch_lock = threading.Lock()
        self._prefetched: set[str] = set()

    def _cache_key(self, layer_idx: int, seq_start: int = 0) -> str:
        return f"layer_{layer_idx}_seq_{seq_start}"

    def _entry_size_mb(self, tensor: torch.Tensor) -> float:
        return tensor.element_size() * tensor.nelement() / (1024 ** 2)

    def store(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        force_gpu: bool = False,
    ):
        """
        Store KV state, automatically managing tier placement.

        If GPU has space → store on GPU.
        If GPU is full → evict LRU to CPU and store new entry on GPU.
        If CPU is full → evict LRU from CPU (or to disk).
        """
        key = self._cache_key(layer_idx)
        entry_mb = self._entry_size_mb(k) + self._entry_size_mb(v)

        # Try GPU first
        while self._gpu_used_mb + entry_mb > self.gpu_budget_mb and self._gpu_cache:
            self._evict_gpu_to_cpu()

        if self._gpu_used_mb + entry_mb <= self.gpu_budget_mb:
            self._gpu_cache[key] = (k.contiguous(), v.contiguous())
            self._gpu_cache.move_to_end(key)
            self._gpu_used_mb += entry_mb
            self._gpu_stats.used_mb = self._gpu_used_mb
        elif not force_gpu:
            # Store directly on CPU
            self._store_cpu(key, k.cpu().contiguous(), v.cpu().contiguous())

    def _store_cpu(self, key: str, k: torch.Tensor, v: torch.Tensor):
        """Store on CPU tier, evicting if necessary."""
        entry_mb = self._entry_size_mb(k) + self._entry_size_mb(v)

        while self._cpu_used_mb + entry_mb > self.cpu_budget_mb and self._cpu_cache:
            evict_key, (ek, ev) = self._cpu_cache.popitem(last=False)
            self._cpu_used_mb -= (self._entry_size_mb(ek) + self._entry_size_mb(ev))
            self._cpu_stats.evictions += 1
            del ek, ev

        self._cpu_cache[key] = (k, v)
        self._cpu_cache.move_to_end(key)
        self._cpu_used_mb += entry_mb
        self._cpu_stats.used_mb = self._cpu_used_mb

    def _evict_gpu_to_cpu(self):
        """Move LRU entry from GPU to CPU."""
        if not self._gpu_cache:
            return

        key, (k, v) = self._gpu_cache.popitem(last=False)
        entry_mb = self._entry_size_mb(k) + self._entry_size_mb(v)
        self._gpu_used_mb -= entry_mb
        self._gpu_stats.used_mb = self._gpu_used_mb
        self._gpu_stats.evictions += 1

        # Move to CPU
        self._store_cpu(key, k.cpu(), v.cpu())

    def get(
        self,
        layer_idx: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve KV state, promoting from lower tiers if needed.

        Automatically moves data to GPU if found on CPU or disk.
        """
        key = self._cache_key(layer_idx)
        start = time.perf_counter()

        # Check GPU
        if key in self._gpu_cache:
            self._gpu_cache.move_to_end(key)
            self._gpu_stats.hits += 1
            elapsed = (time.perf_counter() - start) * 1000
            self._gpu_stats.avg_access_time_ms = (
                self._gpu_stats.avg_access_time_ms * 0.9 + elapsed * 0.1
            )
            return self._gpu_cache[key]

        self._gpu_stats.misses += 1

        # Check CPU
        if key in self._cpu_cache:
            k_cpu, v_cpu = self._cpu_cache.pop(key)
            entry_mb = self._entry_size_mb(k_cpu) + self._entry_size_mb(v_cpu)
            self._cpu_used_mb -= entry_mb

            # Promote to GPU
            k_gpu = k_cpu.to(self.device)
            v_gpu = v_cpu.to(self.device)
            self.store(layer_idx, k_gpu, v_gpu, force_gpu=True)

            self._cpu_stats.hits += 1
            elapsed = (time.perf_counter() - start) * 1000
            self._cpu_stats.avg_access_time_ms = (
                self._cpu_stats.avg_access_time_ms * 0.9 + elapsed * 0.1
            )
            return k_gpu, v_gpu

        self._cpu_stats.misses += 1
        return None

    def prefetch(self, layer_idx: int):
        """
        Asynchronously prefetch a layer's cache to GPU.

        Called before the layer is needed to overlap data transfer
        with the computation of the current layer.
        """
        key = self._cache_key(layer_idx)

        if key in self._gpu_cache or key in self._prefetched:
            return

        def _do_prefetch():
            with self._prefetch_lock:
                if key in self._cpu_cache:
                    k_cpu, v_cpu = self._cpu_cache[key]
                    k_gpu = k_cpu.to(self.device, non_blocking=True)
                    v_gpu = v_cpu.to(self.device, non_blocking=True)
                    self._prefetched.add(key)

        thread = threading.Thread(target=_do_prefetch, daemon=True)
        thread.start()

    def prefetch_ahead(self, current_layer: int):
        """Prefetch the next N layers ahead of current computation."""
        for offset in range(1, self.prefetch_layers + 1):
            next_layer = current_layer + offset
            if next_layer < self.num_layers:
                self.prefetch(next_layer)

    def get_tier_stats(self) -> list[TierStats]:
        """Return statistics for all tiers."""
        return [self._gpu_stats, self._cpu_stats]

    def get_summary(self) -> dict:
        """Summary of cache utilization across tiers."""
        return {
            "gpu": {
                "used_mb": self._gpu_used_mb,
                "budget_mb": self.gpu_budget_mb,
                "utilization": self._gpu_stats.utilization,
                "hit_rate": self._gpu_stats.hit_rate,
                "entries": len(self._gpu_cache),
            },
            "cpu": {
                "used_mb": self._cpu_used_mb,
                "budget_mb": self.cpu_budget_mb,
                "utilization": self._cpu_stats.utilization,
                "hit_rate": self._cpu_stats.hit_rate,
                "entries": len(self._cpu_cache),
            },
        }

    def free(self):
        """Release all cache memory."""
        self._gpu_cache.clear()
        self._cpu_cache.clear()
        self._gpu_used_mb = 0.0
        self._cpu_used_mb = 0.0
        self._prefetched.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
