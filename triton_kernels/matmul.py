# TokenForge GPU-Accelerated LLM Inference Platform
"""
Matrix multiplication implementation using OpenAI Triton.
Demonstrates blocked matmul for optimal performance, rivaling cuBLAS.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_ptrs = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=rn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * rm[:, None] + stride_cn * rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    return c

def benchmark_triton_matmul():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    sizes = [128, 256, 512, 1024, 2048]

    table = Table(title="Triton Matmul Benchmark (FP16)")
    table.add_column("Size", justify="right")
    table.add_column("Triton (ms)", justify="right", style="green")
    table.add_column("cuBLAS (ms)", justify="right", style="yellow")
    
    for size in sizes:
        a = torch.randn((size, size), device='cuda', dtype=torch.float16)
        b = torch.randn((size, size), device='cuda', dtype=torch.float16)
        
        for _ in range(5):
            matmul_triton(a, b)
            torch.mm(a, b)
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(20):
            matmul_triton(a, b)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 20
        
        start.record()
        for _ in range(20):
            torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 20
        
        table.add_row(f"{size}x{size}", f"{triton_ms:.4f}", f"{torch_ms:.4f}")
        
    console.print(table)

if __name__ == "__main__":
    a = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    b = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    c_triton = matmul_triton(a, b)
    c_torch = torch.mm(a, b)
    assert torch.allclose(c_triton, c_torch, atol=1e-2), "Triton matmul mismatch!"
    print("Triton matmul: PASS")
    benchmark_triton_matmul()
