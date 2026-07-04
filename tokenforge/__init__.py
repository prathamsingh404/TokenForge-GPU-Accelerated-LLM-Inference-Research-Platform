"""
TokenForge — The open-source playground for studying, benchmarking,
profiling, and optimizing LLM inference.

Public API:
    from tokenforge import TokenForgeModel, Benchmark, WorkloadConfig
"""

__version__ = "0.2.0"
__project__ = "TokenForge"

# Lazy imports to avoid heavy torch dependency on package load
def __getattr__(name):
    if name == "TokenForgeModel":
        from tokenforge.model import TokenForgeModel
        return TokenForgeModel
    if name == "Benchmark":
        from tokenforge.benchmark import Benchmark
        return Benchmark
    if name == "WorkloadConfig":
        from tokenforge.workloads.generator import WorkloadConfig
        return WorkloadConfig
    raise AttributeError(f"module 'tokenforge' has no attribute {name!r}")


__all__ = [
    "TokenForgeModel",
    "Benchmark",
    "WorkloadConfig",
    "__version__",
]
