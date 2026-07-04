# TokenForge GPU-Accelerated LLM Inference Platform
"""
Vector addition CUDA kernel.

The "hello world" of GPU programming. Demonstrates the fundamental
CUDA concepts: threads, blocks, grid, and how work is distributed
across GPU cores.

Grid layout:
    gridDim.x  = ceil(N / blockDim.x)  — number of thread blocks
    blockDim.x = 256                    — threads per block
    Each thread computes one element: C[i] = A[i] + B[i]
"""

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input sizes must match");

    auto c = torch::empty_like(a);
    int n = a.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}
"""

CPP_SOURCE = """
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);
"""

# Lazy compilation — compiled on first import
_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="vector_add",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["vector_add_cuda"],
            verbose=False,
        )
    return _module


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two vectors using a custom CUDA kernel."""
    mod = _get_module()
    return mod.vector_add_cuda(a.float().cuda(), b.float().cuda())


def benchmark_vector_add(sizes: list[int] = None, num_runs: int = 100):
    """Compare custom kernel vs PyTorch native vs CPU."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    sizes = sizes or [1024, 65536, 1_000_000, 10_000_000, 100_000_000]

    table = Table(title="Vector Addition Benchmark")
    table.add_column("Size", style="cyan", justify="right")
    table.add_column("Custom CUDA (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    for n in sizes:
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")

        # Warmup
        for _ in range(5):
            vector_add(a, b)
            a + b

        # Custom kernel
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            vector_add(a, b)
        end.record()
        torch.cuda.synchronize()
        custom_ms = start.elapsed_time(end) / num_runs

        # PyTorch
        start.record()
        for _ in range(num_runs):
            a + b
        end.record()
        torch.cuda.synchronize()
        pytorch_ms = start.elapsed_time(end) / num_runs

        speedup = pytorch_ms / max(custom_ms, 1e-6)

        table.add_row(
            f"{n:>12,}",
            f"{custom_ms:.4f}",
            f"{pytorch_ms:.4f}",
            f"{speedup:.2f}x",
        )

    console.print(table)


if __name__ == "__main__":
    # Quick test
    a = torch.randn(1000, device="cuda")
    b = torch.randn(1000, device="cuda")
    c = vector_add(a, b)
    expected = a + b
    assert torch.allclose(c, expected, atol=1e-5), "Vector add kernel mismatch!"
    print("Vector add kernel: PASS")

    benchmark_vector_add()
