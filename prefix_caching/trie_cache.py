"""
Trie-based prefix cache for token sequences.

When many requests share a common prompt prefix (e.g., a system
prompt or shared context), we can cache the computed KV states
for that prefix and reuse them. This avoids redundant prefill
computation.

The trie allows efficient longest-prefix matching to find the
best cached state for any incoming request.
"""

import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """Node in the prefix trie. Each node corresponds to a token."""
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    cache_key: Optional[str] = None  # If set, KV cache is available here
    depth: int = 0
    access_count: int = 0
    last_access: float = 0.0


@dataclass
class CacheEntry:
    """A cached KV state for a prefix."""
    key: str
    token_prefix: list[int]
    kv_cache: object  # The actual cache tensors
    prefix_length: int
    creation_time: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0


class PrefixTrieCache:
    """
    Trie data structure for prefix matching + LRU eviction.

    Usage:
        cache = PrefixTrieCache(max_entries=100)

        # Store a cached prefix
        cache.insert(token_ids, kv_cache_tensors)

        # Look up the longest matching prefix for a new request
        match = cache.longest_prefix_match(new_token_ids)
        if match:
            # Reuse match.kv_cache, only compute remaining tokens
            ...
    """

    def __init__(self, max_entries: int = 100, max_memory_mb: float = 1024):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)

        self._entries: dict[str, CacheEntry] = {}
        self._total_bytes = 0

        # Stats
        self.hits = 0
        self.misses = 0

    def insert(
        self,
        token_ids: list[int],
        kv_cache: object,
        size_bytes: int = 0,
    ) -> str:
        """Insert a prefix and its KV cache into the trie."""
        # Evict if necessary
        while (
            len(self._entries) >= self.max_entries
            or (self.max_memory_bytes > 0 and self._total_bytes + size_bytes > self.max_memory_bytes)
        ) and self._entries:
            self._evict_lru()

        # Walk/build trie path
        node = self.root
        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                node.children[token_id] = TrieNode(depth=i + 1)
            node = node.children[token_id]

        # Create cache entry
        cache_key = f"prefix_{len(self._entries)}_{len(token_ids)}"
        node.cache_key = cache_key

        now = time.monotonic()
        entry = CacheEntry(
            key=cache_key,
            token_prefix=list(token_ids),
            kv_cache=kv_cache,
            prefix_length=len(token_ids),
            creation_time=now,
            last_access=now,
            size_bytes=size_bytes,
        )
        self._entries[cache_key] = entry
        self._total_bytes += size_bytes

        return cache_key

    def longest_prefix_match(
        self, token_ids: list[int]
    ) -> Optional[CacheEntry]:
        """
        Find the longest cached prefix that matches the beginning
        of the given token sequence.
        """
        node = self.root
        best_key = None
        best_depth = 0

        for i, token_id in enumerate(token_ids):
            if token_id not in node.children:
                break
            node = node.children[token_id]
            if node.cache_key is not None:
                best_key = node.cache_key
                best_depth = i + 1

        if best_key and best_key in self._entries:
            entry = self._entries[best_key]
            entry.access_count += 1
            entry.last_access = time.monotonic()
            self.hits += 1
            return entry

        self.misses += 1
        return None

    def _evict_lru(self):
        """Evict the least recently used entry."""
        if not self._entries:
            return

        lru_key = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].last_access,
        )

        entry = self._entries.pop(lru_key)
        self._total_bytes -= entry.size_bytes

        # Remove from trie
        self._remove_trie_path(entry.token_prefix)

    def _remove_trie_path(self, token_ids: list[int]):
        """Remove cache_key marker from the trie node."""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                return
            node = node.children[tid]
        node.cache_key = None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def memory_used_mb(self) -> float:
        return self._total_bytes / (1024 ** 2)

    def get_stats(self) -> dict:
        return {
            "entries": self.num_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "memory_mb": self.memory_used_mb,
        }

    def clear(self):
        self.root = TrieNode()
        self._entries.clear()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
