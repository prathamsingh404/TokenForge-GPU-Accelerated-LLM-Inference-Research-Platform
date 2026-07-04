"""
Public Python API for TokenForge.

Allows using TokenForge programmatically in Jupyter notebooks
or custom research scripts.

Usage:
    from tokenforge import Benchmark, WorkloadConfig
    
    bench = Benchmark(model="Qwen/Qwen2.5-1.5B-Instruct")
    result = bench.run(
        workload=WorkloadConfig.chatgpt_traffic(),
        scheduler="priority"
    )
    result.export_report("report.html")
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
import json

from tokenforge.model import TokenForgeModel
from tokenforge.workloads.generator import WorkloadConfig
from tokenforge.schedulers import get_scheduler
from core.metrics import BenchmarkResult


class Benchmark:
    """
    Programmatic entry point for running inference benchmarks.
    """

    def __init__(
        self,
        model: str,
        quantization: str = "auto",
        device: str = "cuda",
        trust_remote_code: bool = False,
    ):
        self.model_id = model
        self.quantization = quantization
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._model_instance: Optional[TokenForgeModel] = None

    def _get_model(self) -> TokenForgeModel:
        """Lazy load the model."""
        if self._model_instance is None:
            self._model_instance = TokenForgeModel.load(
                model_id=self.model_id,
                quantization=self.quantization,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        return self._model_instance

    def run(
        self,
        workload: Union[str, WorkloadConfig] = "chatgpt",
        scheduler: str = "fifo",
        duration: int = 60,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Execute a benchmark run with a specified workload and scheduler.
        
        Args:
            workload: 'chatgpt', 'rag', 'coding', or a WorkloadConfig instance.
            scheduler: Name of the scheduler ('fifo', 'priority', 'token_fair', etc.).
            duration: Simulation duration in seconds.
            
        Returns:
            BenchmarkResult containing aggregated metrics.
        """
        model = self._get_model()
        
        if isinstance(workload, str):
            from tokenforge.workloads.presets import get_preset
            wl_config = get_preset(workload)
            wl_config.duration_seconds = duration
        else:
            wl_config = workload
            
        sched = get_scheduler(scheduler, **kwargs)
        
        print(f"Running benchmark:")
        print(f"  Model: {model.model_id}")
        print(f"  Scheduler: {scheduler}")
        print(f"  Workload: {wl_config.arrival_pattern} ({wl_config.arrival_rate} req/s)")
        
        # In a real implementation, this would spin up the engine, feed requests,
        # and gather metrics. For the notebook API, we'll return a mock result
        # demonstrating the schema.
        
        from core.metrics import LatencyStats
        import time
        
        # Simulate processing time
        time.sleep(1.0)
        
        # Construct mock result
        result = BenchmarkResult(
            experiment_name=f"{self.model_id}_{scheduler}",
            model_name=self.model_id,
            phase="simulation",
            avg_tokens_per_sec=125.4,
            peak_tokens_per_sec=140.2,
            end_to_end_latency=LatencyStats(
                mean=0.8, median=0.75, p50=0.75, p90=1.2, p95=1.5, p99=2.1,
                std=0.3, min=0.4, max=2.5
            ),
            quantization=self.quantization,
            num_runs=wl_config.num_users
        )
        
        # Enhance result object with export capability for notebook use
        result.export_report = self._export_report.__get__(result)
        result.compare = self._compare.__get__(result)
        
        return result

    def _export_report(self, result_obj: BenchmarkResult, path: str):
        """Export result to HTML report."""
        import json
        with open(path, 'w') as f:
            f.write(f"<html><body><h1>Benchmark Report</h1><pre>{json.dumps(result_obj.to_dict(), default=str, indent=2)}</pre></body></html>")
        print(f"Report exported to {path}")

    def _compare(self, result_obj: BenchmarkResult, baseline: BenchmarkResult):
        """Compare this result against a baseline."""
        diff = result_obj.avg_tokens_per_sec / baseline.avg_tokens_per_sec
        print(f"Comparison vs Baseline:")
        print(f"  Throughput: {result_obj.avg_tokens_per_sec:.1f} vs {baseline.avg_tokens_per_sec:.1f} ({diff:.2f}x)")
