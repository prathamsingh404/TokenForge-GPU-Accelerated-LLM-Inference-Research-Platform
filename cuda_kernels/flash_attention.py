"""
Flash Attention style CUDA kernel.

Simplified implementation of the key insight from the Flash Attention
paper (Dao et al., 2022): compute attention in tiles to minimize
HBM (global memory) access. Instead of materializing the full
N×N attention matrix, we accumulate results block by block using
online softmax.

This is a teaching implementation — real Flash Attention has more
sophisticated tiling, vectorized loads, and warp-level primitives.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_SIZE 32

// Simplified Flash Attention: processes Q and K/V in blocks.
// For each block of Q rows, iterates over blocks of K/V columns,
// computing partial attention scores and accumulating the output
// without materializing the full attention matrix.
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,   // (seq_len, head_dim)
    const float* __restrict__ K,   // (seq_len, head_dim)
    const float* __restrict__ V,   // (seq_len, head_dim)
    float* __restrict__ O,         // (seq_len, head_dim)
    int seq_len,
    int head_dim,
    float scale
) {
    // Each block handles one row of Q (one query position)
    int q_idx = blockIdx.x;
    int d = threadIdx.x;

    if (q_idx >= seq_len || d >= head_dim) return;

    // Load this query row
    float q_val = Q[q_idx * head_dim + d];

    // Online softmax accumulators
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc = 0.0f;  // Accumulated output for this (q_idx, d)

    // Iterate over K/V in blocks
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_SIZE) {
        int kv_end = min(kv_start + BLOCK_SIZE, seq_len);

        for (int kv_idx = kv_start; kv_idx < kv_end; kv_idx++) {
            // Compute dot product Q[q_idx] · K[kv_idx]
            // Each thread handles one dimension; need reduction across threads
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += Q[q_idx * head_dim + dd] * K[kv_idx * head_dim + dd];
            }
            dot *= scale;

            // Online softmax update
            float old_max = running_max;
            running_max = fmaxf(running_max, dot);

            float exp_old = expf(old_max - running_max);
            float exp_new = expf(dot - running_max);

            // Rescale running sum and accumulator
            running_sum = running_sum * exp_old + exp_new;
            acc = acc * exp_old + exp_new * V[kv_idx * head_dim + d];
        }
    }

    // Final normalization
    O[q_idx * head_dim + d] = acc / running_sum;
}

torch::Tensor flash_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.dim() == 2 && K.dim() == 2 && V.dim() == 2, "Inputs must be 2D");
    int seq_len = Q.size(0);
    int head_dim = Q.size(1);

    float scale = 1.0f / sqrtf((float)head_dim);
    auto O = torch::zeros_like(Q);

    // One block per query position, head_dim threads per block
    int threads = min(head_dim, 1024);
    flash_attention_kernel<<<seq_len, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        seq_len, head_dim, scale
    );

    return O;
}
"""

CPP_SOURCE = """
torch::Tensor flash_attention_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="flash_attention",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["flash_attention_cuda"],
            verbose=False,
        )
    return _module


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Simplified Flash Attention with online softmax."""
    mod = _get_module()
    return mod.flash_attention_cuda(
        Q.float().cuda(), K.float().cuda(), V.float().cuda()
    )


def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Standard attention for comparison (materializes full N×N matrix)."""
    scale = Q.shape[-1] ** -0.5
    scores = torch.mm(Q, K.t()) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.mm(attn, V)


def benchmark_flash_attention(sizes: list[tuple] = None, num_runs: int = 20):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    sizes = sizes or [(64, 64), (128, 64), (256, 64), (512, 64), (128, 128)]

    table = Table(title="Flash Attention Benchmark (single head)")
    table.add_column("seq×dim", style="cyan")
    table.add_column("Flash (ms)", justify="right", style="green")
    table.add_column("Standard (ms)", justify="right", style="yellow")
    table.add_column("Ratio", justify="right")

    for seq_len, head_dim in sizes:
        Q = torch.randn(seq_len, head_dim, device="cuda")
        K = torch.randn(seq_len, head_dim, device="cuda")
        V = torch.randn(seq_len, head_dim, device="cuda")

        for _ in range(3):
            flash_attention(Q, K, V)
            standard_attention(Q, K, V)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            flash_attention(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        flash_ms = start.elapsed_time(end) / num_runs

        start.record()
        for _ in range(num_runs):
            standard_attention(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        std_ms = start.elapsed_time(end) / num_runs

        ratio = flash_ms / max(std_ms, 1e-6)
        table.add_row(f"{seq_len}×{head_dim}", f"{flash_ms:.4f}", f"{std_ms:.4f}", f"{ratio:.2f}x")

    console.print(table)


if __name__ == "__main__":
    # Correctness check
    Q = torch.randn(32, 64, device="cuda")
    K = torch.randn(32, 64, device="cuda")
    V = torch.randn(32, 64, device="cuda")

    flash_out = flash_attention(Q, K, V)
    std_out = standard_attention(Q, K, V)

    assert torch.allclose(flash_out, std_out, atol=1e-3), "Flash attention mismatch!"
    print("Flash Attention kernel: PASS")

    benchmark_flash_attention()
