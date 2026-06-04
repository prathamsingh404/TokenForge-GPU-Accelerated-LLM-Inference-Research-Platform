"""
Benchmark suite for all Triton kernels.
"""

from rich.console import Console

console = Console()

def run_all_triton_benchmarks():
    console.print("\n[bold magenta]{'='*60}[/]")
    console.print("[bold magenta]Triton Kernel Benchmark Suite[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # Vector Addition
    console.print("[bold cyan]1. Vector Addition[/]\n")
    try:
        from triton_kernels.vector_add import benchmark_triton_add
        benchmark_triton_add()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # Matrix Multiplication
    console.print("\n[bold cyan]2. Matrix Multiplication[/]\n")
    try:
        from triton_kernels.matmul import benchmark_triton_matmul
        benchmark_triton_matmul()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # Softmax
    console.print("\n[bold cyan]3. Softmax[/]\n")
    try:
        from triton_kernels.softmax import benchmark_triton_softmax
        benchmark_triton_softmax()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # LayerNorm
    console.print("\n[bold cyan]4. LayerNorm[/]\n")
    try:
        from triton_kernels.layernorm import benchmark_triton_layernorm
        benchmark_triton_layernorm()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    # Fused Attention
    console.print("\n[bold cyan]5. Fused Attention[/]\n")
    try:
        from triton_kernels.fused_attention import benchmark_triton_attention
        benchmark_triton_attention()
    except Exception as e:
        console.print(f"[red]Failed: {e}[/]")

    console.print(f"\n[bold green]{'='*60}[/]")
    console.print("[bold green]All Triton benchmarks complete.[/]")
    console.print(f"[bold green]{'='*60}[/]")

if __name__ == "__main__":
    run_all_triton_benchmarks()
