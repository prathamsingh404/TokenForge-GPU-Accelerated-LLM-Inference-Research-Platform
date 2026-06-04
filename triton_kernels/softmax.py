"""
Softmax implementation using OpenAI Triton.

Triton allows writing high-performance GPU kernels in Python.
It handles memory coalescing, shared memory management, and
scheduling automatically, making it much easier than raw CUDA.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for softmax.
    Each program instance processes one row of the input matrix.
    """
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)

    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > n_cols
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)

    # Numerator
    numerator = tl.exp(row_minus_max)

    # Denominator
    denominator = tl.sum(numerator, axis=0)

    # Softmax
    softmax_output = numerator / denominator

    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Apply softmax along the last dimension using Triton."""
    # We need a 2D tensor for the kernel, so we flatten everything except the last dim
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    # The block size must be a power of two greater than or equal to n_cols
    # triton.next_power_of_2 is useful here
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output
    y_2d = torch.empty_like(x_2d)

    # Enqueue kernel. The 1D grid runs one program per row.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    softmax_kernel[(n_rows,)](
        y_2d,
        x_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_2d.view_as(x)


def benchmark_triton_softmax(sizes: list[tuple] = None, num_runs: int = 100):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    sizes = sizes or [(128, 128), (256, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]

    table = Table(title="Triton vs PyTorch Softmax Benchmark")
    table.add_column("Shape", style="cyan")
    table.add_column("Triton (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    for rows, cols in sizes:
        x = torch.randn(rows, cols, device="cuda")

        # Warmup
        for _ in range(10):
            softmax_triton(x)
            torch.softmax(x, dim=-1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Triton
        start.record()
        for _ in range(num_runs):
            softmax_triton(x)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / num_runs

        # PyTorch
        start.record()
        for _ in range(num_runs):
            torch.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        pytorch_ms = start.elapsed_time(end) / num_runs

        speedup = pytorch_ms / max(triton_ms, 1e-6)

        table.add_row(
            f"{rows}×{cols}",
            f"{triton_ms:.4f}",
            f"{pytorch_ms:.4f}",
            f"{speedup:.2f}x",
        )

    console.print(table)


if __name__ == "__main__":
    # Correctness check
    x = torch.randn(64, 128, device="cuda")
    result = softmax_triton(x)
    expected = torch.softmax(x, dim=-1)

    assert torch.allclose(result, expected, atol=1e-4), "Triton softmax mismatch!"
    print("Triton softmax: PASS")

    benchmark_triton_softmax()
