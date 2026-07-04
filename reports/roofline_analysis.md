# Roofline Analysis

The Roofline model is a visual tool used to evaluate the theoretical upper bound of a given kernel's performance on specific hardware (e.g., RTX 5050).

## Arithmetic Intensity
Defined as:
$$ \text{Arithmetic Intensity (FLOPs/Byte)} = \frac{\text{Total Floating Point Operations}}{\text{Total Memory Bytes Transferred}} $$

## The Two Regimes
1. **Memory-Bound**: The kernel does very little math per byte loaded (e.g., Vector Addition, LayerNorm, autoregressive decoding). Performance is capped by VRAM Bandwidth (GB/s).
2. **Compute-Bound**: The kernel does massive amounts of math per byte loaded (e.g., Large Matrix Multiplication, Prefill attention). Performance is capped by Peak Compute (TFLOPs).

By plotting our custom CUDA and Triton kernels on a log-log Roofline plot, we can determine whether further optimizations should focus on memory coalescing (if memory bound) or instruction-level parallelism (if compute bound).


<!-- TokenForge Platform -->
