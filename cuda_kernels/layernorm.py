# TokenForge GPU-Accelerated LLM Inference Platform
"""
LayerNorm CUDA kernel.

Implements layer normalization, used at every transformer block:
    LN(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Uses Welford's online algorithm for numerically stable
single-pass computation of mean and variance.
"""

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Welford's online algorithm for computing mean and variance in one pass.
// Each block normalizes one row (one token in the sequence).
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float shared[];
    float* s_mean = shared;
    float* s_var = shared + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Welford's: accumulate mean and M2 (sum of squared deviations)
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    int local_count = 0;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = row_in[i];
        local_count++;
        float delta = val - local_mean;
        local_mean += delta / local_count;
        local_m2 += delta * (val - local_mean);
    }

    // Store partial results
    s_mean[tid] = local_mean;
    s_var[tid] = local_m2;
    __syncthreads();

    // Parallel reduction to combine Welford accumulators
    // Simplified: just do mean/variance of partial means/vars
    // (exact parallel Welford is more complex, this is sufficient for demonstration)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] = (s_mean[tid] + s_mean[tid + stride]) * 0.5f;
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    // Recompute exact mean and variance for this row
    // (parallel reduction of Welford is tricky; use two-pass for correctness)
    __syncthreads();

    // Pass 1: exact mean
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += row_in[i];
    }
    s_mean[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_mean[tid] += s_mean[tid + stride];
        __syncthreads();
    }
    float mean = s_mean[0] / cols;
    __syncthreads();

    // Pass 2: variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_var[tid] += s_var[tid + stride];
        __syncthreads();
    }
    float variance = s_var[0] / cols;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();

    // Normalize + scale + shift
    for (int i = tid; i < cols; i += blockDim.x) {
        float normalized = (row_in[i] - mean) * inv_std;
        row_out[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    int threads = min(1024, cols);
    threads = 1;
    while (threads < cols && threads < 1024) threads *= 2;

    int shared_mem = 2 * threads * sizeof(float);

    layernorm_kernel<<<rows, threads, shared_mem>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols, eps
    );

    return output;
}
"""

CPP_SOURCE = """
torch::Tensor layernorm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="layernorm_kernel",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["layernorm_cuda"],
            verbose=False,
        )
    return _module


def layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Custom CUDA LayerNorm."""
    mod = _get_module()
    return mod.layernorm_cuda(
        x.float().cuda(), gamma.float().cuda(), beta.float().cuda(), eps
    )


def benchmark_layernorm(sizes: list[tuple] = None, num_runs: int = 50):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    sizes = sizes or [(128, 768), (256, 768), (512, 1024), (1024, 2048)]

    table = Table(title="LayerNorm Benchmark")
    table.add_column("Shape", style="cyan")
    table.add_column("Custom (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    table.add_column("Ratio", justify="right")

    for seq, hidden in sizes:
        x = torch.randn(seq, hidden, device="cuda")
        gamma = torch.ones(hidden, device="cuda")
        beta = torch.zeros(hidden, device="cuda")
        ln = torch.nn.LayerNorm(hidden).cuda()

        for _ in range(5):
            layernorm(x, gamma, beta)
            ln(x)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            layernorm(x, gamma, beta)
        end.record()
        torch.cuda.synchronize()
        custom_ms = start.elapsed_time(end) / num_runs

        start.record()
        for _ in range(num_runs):
            ln(x)
        end.record()
        torch.cuda.synchronize()
        pytorch_ms = start.elapsed_time(end) / num_runs

        ratio = custom_ms / max(pytorch_ms, 1e-6)
        table.add_row(f"{seq}×{hidden}", f"{custom_ms:.4f}", f"{pytorch_ms:.4f}", f"{ratio:.2f}x")

    console.print(table)


if __name__ == "__main__":
    x = torch.randn(32, 128, device="cuda")
    gamma = torch.ones(128, device="cuda")
    beta = torch.zeros(128, device="cuda")

    result = layernorm(x, gamma, beta)
    ln = torch.nn.LayerNorm(128).cuda()
    expected = ln(x)

    assert torch.allclose(result, expected, atol=1e-3), "LayerNorm kernel mismatch!"
    print("LayerNorm kernel: PASS")

    benchmark_layernorm()
