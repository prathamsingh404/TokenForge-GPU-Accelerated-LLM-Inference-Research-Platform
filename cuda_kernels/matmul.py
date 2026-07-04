# TokenForge GPU-Accelerated LLM Inference Platform
"""
Matrix multiplication CUDA kernels.

Implements three versions with increasing optimization:
1. Naive — one thread per output element, direct global memory access
2. Tiled — shared memory tiling to exploit data reuse
3. cuBLAS — via torch.mm, the gold standard

Comparing these demonstrates why memory hierarchy matters.
"""

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// --- Naive Matrix Multiply ---
// Each thread computes one element of C by iterating over the shared dimension.
// Every access goes to global memory (slow).
__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --- Tiled Matrix Multiply ---
// Uses shared memory to load tiles of A and B, reducing global memory traffic.
// Each tile fits in shared memory (fast SRAM on the SM).
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_naive_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K
    );
    return C;
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, N, K
    );
    return C;
}
"""

CPP_SOURCE = """
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="matmul_kernels",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["matmul_naive", "matmul_tiled"],
            verbose=False,
        )
    return _module


def matmul_naive(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    mod = _get_module()
    return mod.matmul_naive(A.float().cuda(), B.float().cuda())


def matmul_tiled(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    mod = _get_module()
    return mod.matmul_tiled(A.float().cuda(), B.float().cuda())


def benchmark_matmul(sizes: list[int] = None, num_runs: int = 20):
    """Compare naive, tiled, and cuBLAS matrix multiplication."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    sizes = sizes or [128, 256, 512, 1024, 2048]

    table = Table(title="Matrix Multiplication Benchmark (FP32)")
    table.add_column("Size", style="cyan", justify="right")
    table.add_column("Naive (ms)", justify="right")
    table.add_column("Tiled (ms)", justify="right", style="green")
    table.add_column("cuBLAS (ms)", justify="right", style="yellow")
    table.add_column("Tiled Speedup", justify="right")
    table.add_column("cuBLAS Speedup", justify="right", style="magenta")

    for M in sizes:
        A = torch.randn(M, M, device="cuda")
        B = torch.randn(M, M, device="cuda")

        # Warmup
        for _ in range(3):
            matmul_naive(A, B)
            matmul_tiled(A, B)
            torch.mm(A, B)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Naive
        start.record()
        for _ in range(num_runs):
            matmul_naive(A, B)
        end.record()
        torch.cuda.synchronize()
        naive_ms = start.elapsed_time(end) / num_runs

        # Tiled
        start.record()
        for _ in range(num_runs):
            matmul_tiled(A, B)
        end.record()
        torch.cuda.synchronize()
        tiled_ms = start.elapsed_time(end) / num_runs

        # cuBLAS
        start.record()
        for _ in range(num_runs):
            torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        cublas_ms = start.elapsed_time(end) / num_runs

        tiled_speedup = naive_ms / max(tiled_ms, 1e-6)
        cublas_speedup = naive_ms / max(cublas_ms, 1e-6)

        table.add_row(
            f"{M}×{M}",
            f"{naive_ms:.3f}",
            f"{tiled_ms:.3f}",
            f"{cublas_ms:.3f}",
            f"{tiled_speedup:.1f}x",
            f"{cublas_speedup:.1f}x",
        )

    console.print(table)


if __name__ == "__main__":
    # Correctness check
    A = torch.randn(64, 64, device="cuda")
    B = torch.randn(64, 64, device="cuda")

    c_naive = matmul_naive(A, B)
    c_tiled = matmul_tiled(A, B)
    c_ref = torch.mm(A, B)

    assert torch.allclose(c_naive, c_ref, atol=1e-3), "Naive matmul mismatch!"
    assert torch.allclose(c_tiled, c_ref, atol=1e-3), "Tiled matmul mismatch!"
    print("Matrix multiply kernels: PASS")

    benchmark_matmul()
