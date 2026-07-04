"""
Plugin abstract base classes.

Users implement these to extend TokenForge with custom components.

Example:
    class MyScheduler(SchedulerPlugin):
        name = "my_scheduler"
        
        def schedule(self, waiting, active, max_batch):
            # Custom scheduling logic
            ...

    PluginRegistry.register(MyScheduler)
"""

from abc import ABC, abstractmethod
from typing import Optional


class PluginMeta:
    """Base metadata for all plugins."""
    name: str = "unnamed"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""


class SchedulerPlugin(PluginMeta, ABC):
    """
    Plugin interface for custom scheduling algorithms.

    Implement schedule() to define how requests are ordered
    and batched for GPU execution.
    """
    plugin_type: str = "scheduler"

    @abstractmethod
    def schedule(self, waiting: list, active: list, max_batch: int) -> dict:
        """
        Select and order requests for the next decode step.

        Args:
            waiting: List of pending SchedulerRequest objects.
            active: List of currently decoding SchedulerRequest objects.
            max_batch: Maximum batch size.

        Returns:
            Dict with 'prefill' and 'decode' lists of SchedulerRequest.
        """
        ...

    def on_token_generated(self, request_id: str):
        """Called when a token is generated for a request."""
        pass

    def on_request_completed(self, request_id: str):
        """Called when a request finishes generation."""
        pass


class CachePlugin(PluginMeta, ABC):
    """
    Plugin interface for custom KV cache strategies.

    Implement allocate/get/update to define how KV states
    are stored, compressed, and evicted.
    """
    plugin_type: str = "cache"

    @abstractmethod
    def allocate(self, num_layers: int, num_heads: int,
                 head_dim: int, max_seq_len: int, **kwargs):
        """Allocate cache memory."""
        ...

    @abstractmethod
    def get(self, layer_idx: int, seq_range: tuple[int, int]):
        """Retrieve cached KV states for a layer and sequence range."""
        ...

    @abstractmethod
    def update(self, layer_idx: int, new_k, new_v):
        """Update cache with new KV pairs."""
        ...

    def evict(self, policy: str = "lru"):
        """Evict entries based on policy. Override for custom eviction."""
        pass

    def compress(self, layer_idx: int, target_dtype=None):
        """Compress cached entries. Override for custom compression."""
        pass

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {}


class AttentionPlugin(PluginMeta, ABC):
    """
    Plugin interface for custom attention mechanisms.

    Implement forward() to define a custom attention computation.
    """
    plugin_type: str = "attention"

    @abstractmethod
    def forward(self, query, key, value, mask=None, **kwargs):
        """
        Compute attention output.

        Args:
            query: Query tensor [batch, heads, seq_len, head_dim]
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask

        Returns:
            Attention output tensor.
        """
        ...

    def supports_flash(self) -> bool:
        """Whether this implementation supports flash attention."""
        return False


class QuantizerPlugin(PluginMeta, ABC):
    """
    Plugin interface for custom quantization strategies.

    Implement quantize/dequantize to define custom precision formats.
    """
    plugin_type: str = "quantizer"

    @abstractmethod
    def quantize(self, tensor, **kwargs):
        """Quantize a tensor to the target precision."""
        ...

    @abstractmethod
    def dequantize(self, quantized_tensor, **kwargs):
        """Dequantize back to floating point."""
        ...

    def get_compression_ratio(self) -> float:
        """Return the theoretical compression ratio."""
        return 1.0
