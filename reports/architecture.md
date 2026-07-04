# Laboratory Architecture

The LLM Inference Laboratory is designed with modularity in mind, separating the concerns of generation, batching, caching, and kernel execution.

## Core Loop
1. **Request Queue**: New inference requests are asynchronously collected and tokenized.
2. **Scheduler (Continuous Batching)**: The scheduler evaluates available KV Cache blocks. If sufficient memory exists, the request enters the "prefill" batch.
3. **Engine Iteration**: The engine executes a single forward pass.
4. **Eviction**: Completed requests are evicted from the batch, and their KV Cache blocks are freed.

## Memory Hierarchy
- The system allocates a static KV Cache block pool upon initialization.
- Tensors are strictly mapped to `torch.float16` by default to preserve VRAM bandwidth.
- Kernel dispatches bypass PyTorch's ATen dispatch overhead where custom Triton/CUDA kernels are injected.


<!-- TokenForge Platform -->
