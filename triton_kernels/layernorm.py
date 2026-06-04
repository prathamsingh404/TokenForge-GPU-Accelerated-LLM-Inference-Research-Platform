"""
LayerNorm implementation using OpenAI Triton.
Computes layer normalization over the last dimension.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    stride_row,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * stride_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols
    
    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    
    # Normalize
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = x_centered * rstd
    
    # Scale and shift
    w = tl.load(w_ptr + col_offsets, mask=mask)
    b = tl.load(b_ptr + col_offsets, mask=mask)
    y = x_hat * w + b
    
    # Store
    out_row_start_ptr = y_ptr + row_idx * stride_row
    tl.store(out_row_start_ptr + col_offsets, y, mask=mask)

def layernorm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y_2d = torch.empty_like(x_2d)
    
    layernorm_kernel[(n_rows,)](
        x_2d, y_2d, weight, bias,
        x_2d.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y_2d.view_as(x)

def benchmark_triton_layernorm():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    sizes = [(128, 768), (256, 1024), (512, 2048), (1024, 4096)]

    table = Table(title="Triton LayerNorm Benchmark")
    table.add_column("Size", justify="right")
    table.add_column("Triton (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    
    for rows, cols in sizes:
        x = torch.randn(rows, cols, device='cuda')
        w = torch.randn(cols, device='cuda')
        b = torch.randn(cols, device='cuda')
        
        # Warmup
        for _ in range(5):
            layernorm_triton(x, w, b)
            torch.nn.functional.layer_norm(x, (cols,), w, b)
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(50):
            layernorm_triton(x, w, b)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 50
        
        start.record()
        for _ in range(50):
            torch.nn.functional.layer_norm(x, (cols,), w, b)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 50
        
        table.add_row(f"{rows}x{cols}", f"{triton_ms:.4f}", f"{torch_ms:.4f}")
        
    console.print(table)

if __name__ == "__main__":
    x = torch.randn(32, 1024, device='cuda')
    w = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    
    y_triton = layernorm_triton(x, w, b)
    y_torch = torch.nn.functional.layer_norm(x, (1024,), w, b)
    assert torch.allclose(y_triton, y_torch, atol=1e-3), "Triton LayerNorm mismatch!"
    print("Triton LayerNorm: PASS")
    benchmark_triton_layernorm()
