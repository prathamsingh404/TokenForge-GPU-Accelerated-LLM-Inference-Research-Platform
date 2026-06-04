"""
Vector addition implementation using OpenAI Triton.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the vector is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Write x + y back to DRAM.
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

def benchmark_triton_add():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    sizes = [1024, 65536, 1_000_000, 10_000_000]

    table = Table(title="Triton Vector Add Benchmark")
    table.add_column("Size", justify="right")
    table.add_column("Triton (ms)", justify="right", style="green")
    table.add_column("PyTorch (ms)", justify="right", style="yellow")
    
    for n in sizes:
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        
        # Warmup
        for _ in range(5):
            add_triton(x, y)
            x + y
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            add_triton(x, y)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 100
        
        start.record()
        for _ in range(100):
            x + y
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 100
        
        table.add_row(f"{n:,}", f"{triton_ms:.4f}", f"{torch_ms:.4f}")
        
    console.print(table)

if __name__ == "__main__":
    x = torch.randn(1000, device='cuda')
    y = torch.randn(1000, device='cuda')
    z = add_triton(x, y)
    assert torch.allclose(z, x + y), "Triton add mismatch!"
    print("Triton vector add: PASS")
    benchmark_triton_add()
