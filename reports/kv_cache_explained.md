# KV Cache Mechanics

During autoregressive generation, a transformer predicts the $N+1$ token by attending to all previous $N$ tokens. Recomputing the Key (K) and Value (V) projections for the entire sequence at every step yields $O(N^2)$ complexity per token.

## The Caching Solution
By storing the K and V tensors for previous tokens, we only compute K and V for the *newest* token, resulting in $O(N)$ complexity.

## Memory Fragmentation
Naively appending to a PyTorch tensor dynamically reallocates memory. Over thousands of tokens across large batch sizes, this fragments the GPU VRAM, causing Out-Of-Memory (OOM) errors even when sufficient total memory exists.

## PagedAttention (vLLM style)
Our `KVCacheManager` implements a simplified PagedAttention. It pre-allocates a continuous memory block divided into pages (e.g., 16 tokens per page). Logical tokens are mapped to physical pages via a block table, eliminating external fragmentation entirely.


<!-- TokenForge Platform -->
