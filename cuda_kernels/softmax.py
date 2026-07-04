# TokenForge GPU-Accelerated LLM Inference Platform
"""
Softmax CUDA kernel.

Implements numerically stable row-wise softmax, the operation at
the heart of attention. Uses the online trick: subtract the row
maximum before exponentiating to prevent overflow.

softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
"""

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Each block handles one row of the input matrix.
// Uses shared memory reduction for the max and sum operations.
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // Step 1: Find row maximum (parallel reduction)
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and partial sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(row_input[i] - row_max);
        row_output[i] = val;  // Store intermediate
        local_sum += val;
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Sum reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    int threads = min(1024, cols);
    // Round up to nearest power of 2 for reduction
    threads = 1;
    while (threads < cols && threads < 1024) threads *= 2;

    int shared_mem = threads * sizeof(float);

    softmax_kernel<<<rows, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}
"""

CPP_SOURCE = "torch::Tensor softmax_cuda(torch::Tensor input);"

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="softmax_kernel",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["softmax_cuda"],
            verbose=False,
        )
    return _module


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax using custom CUDA kernel."""
    mod = _get_module()
    return mod.softmax_cuda(x.float().cuda())


def benchmark_softmax(sizes: list[tuple] = None, num_runs: int = 50):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    sizes = sizes or [(128, 128), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]

    table = Table(title="Softmax Benchmark")
    table.add_column("Shape", style="cyan")
    table.add_column("Custom (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    table.add_column("Ratio", justify="right")

    for rows, cols in sizes:
        x = torch.randn(rows, cols, device="cuda")

        for _ in range(5):
            softmax(x)
            torch.softmax(x, dim=-1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            softmax(x)
        end.record()
        torch.cuda.synchronize()
        custom_ms = start.elapsed_time(end) / num_runs

        start.record()
        for _ in range(num_runs):
            torch.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        pytorch_ms = start.elapsed_time(end) / num_runs

        ratio = custom_ms / max(pytorch_ms, 1e-6)

        table.add_row(
            f"{rows}×{cols}",
            f"{custom_ms:.4f}",
            f"{pytorch_ms:.4f}",
            f"{ratio:.2f}x",
        )

    console.print(table)


if __name__ == "__main__":
    x = torch.randn(64, 128, device="cuda")
    result = softmax(x)
    expected = torch.softmax(x, dim=-1)

    assert torch.allclose(result, expected, atol=1e-4), "Softmax kernel mismatch!"
    assert torch.allclose(result.sum(dim=-1), torch.ones(64, device="cuda"), atol=1e-4), "Rows don't sum to 1!"
    print("Softmax kernel: PASS")

    benchmark_softmax()
