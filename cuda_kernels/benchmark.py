"""
CUDA kernel benchmark suite.

Runs all custom CUDA kernels and compares them against PyTorch
native operations and cuBLAS. Produces a summary report.
"""

from rich.console import Console

console = Console()


def run_all_kernel_benchmarks():
    """Run benchmarks for all custom CUDA kernels."""
    console.print("\n[bold magenta]{'='*60}[/]")
    console.print("[bold magenta]CUDA Kernel Benchmark Suite[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # Vector Addition
    console.print("[bold cyan]1. Vector Addition[/]\n")
    try:
        from cuda_kernels.vector_add import benchmark_vector_add
        benchmark_vector_add()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # Matrix Multiplication
    console.print("\n[bold cyan]2. Matrix Multiplication[/]\n")
    try:
        from cuda_kernels.matmul import benchmark_matmul
        benchmark_matmul()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # Softmax
    console.print("\n[bold cyan]3. Softmax[/]\n")
    try:
        from cuda_kernels.softmax import benchmark_softmax
        benchmark_softmax()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # LayerNorm
    console.print("\n[bold cyan]4. LayerNorm[/]\n")
    try:
        from cuda_kernels.layernorm import benchmark_layernorm
        benchmark_layernorm()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # Flash Attention
    console.print("\n[bold cyan]5. Flash Attention[/]\n")
    try:
        from cuda_kernels.flash_attention import benchmark_flash_attention
        benchmark_flash_attention()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    console.print(f"\n[bold green]{'='*60}[/]")
    console.print("[bold green]All kernel benchmarks complete.[/]")
    console.print(f"[bold green]{'='*60}[/]")


if __name__ == "__main__":
    run_all_kernel_benchmarks()
