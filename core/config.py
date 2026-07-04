# TokenForge GPU-Accelerated LLM Inference Platform
"""
Global configuration for the inference lab.

Centralizes hardware detection, model paths, experiment defaults,
and database settings. Import `LAB_CONFIG` for the active config.
"""

import os
import dataclasses
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "inference_lab.db"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROFILING_DIR = PROJECT_ROOT / "profiling" / "traces"


@dataclasses.dataclass
class HardwareInfo:
    gpu_name: str
    gpu_vram_mb: int
    compute_capability: tuple[int, int]
    cuda_version: str
    sm_count: int
    driver_version: str

    @classmethod
    def detect(cls) -> "HardwareInfo":
        if not torch.cuda.is_available():
            import warnings
            warnings.warn("No CUDA-capable GPU detected. Using mock hardware info for testing.")
            return cls(
                gpu_name="Mock GPU",
                gpu_vram_mb=8192,
                compute_capability=(8, 0),
                cuda_version="0.0",
                sm_count=32,
                driver_version="unknown"
            )

        props = torch.cuda.get_device_properties(0)
        return cls(
            gpu_name=props.name,
            gpu_vram_mb=props.total_memory // (1024 ** 2),
            compute_capability=(props.major, props.minor),
            cuda_version=torch.version.cuda or "unknown",
            sm_count=props.multi_processor_count,
            driver_version=torch.cuda.get_arch_list()[-1] if torch.cuda.get_arch_list() else "unknown",
        )


@dataclasses.dataclass
class ModelConfig:
    """Default models sized for the available VRAM."""

    # Small model for CUDA kernel dev and rapid iteration
    small_model: str = "gpt2"
    small_model_alias: str = "GPT-2 124M"

    # Medium model for inference optimization experiments
    medium_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    medium_model_alias: str = "TinyLlama 1.1B"

    # Large model (quantized only — won't fit in FP16 on 8GB)
    large_model: str = "Qwen/Qwen2.5-7B-Instruct"
    large_model_alias: str = "Qwen2.5 7B"

    # Draft model for speculative decoding
    draft_model: str = "gpt2"
    draft_model_alias: str = "GPT-2 124M"

    # Target model for speculative decoding verification
    target_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target_model_alias: str = "TinyLlama 1.1B"


@dataclasses.dataclass
class BenchmarkDefaults:
    warmup_runs: int = 3
    timed_runs: int = 10
    cooldown_seconds: float = 2.0
    max_new_tokens: int = 128
    batch_sizes: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32]
    )
    sequence_lengths: list[int] = dataclasses.field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    gpu_poll_interval_ms: int = 100


@dataclasses.dataclass
class LabConfig:
    hardware: HardwareInfo
    models: ModelConfig
    benchmarks: BenchmarkDefaults
    db_path: Path
    experiments_dir: Path
    reports_dir: Path
    profiling_dir: Path

    @classmethod
    def build(cls) -> "LabConfig":
        hw = HardwareInfo.detect()
        models = ModelConfig()
        benchmarks = BenchmarkDefaults()

        # Adjust batch sizes if VRAM is tight
        if hw.gpu_vram_mb < 6144:
            benchmarks.batch_sizes = [1, 2, 4, 8]
            benchmarks.max_new_tokens = 64

        # Ensure output directories exist
        for d in [EXPERIMENTS_DIR, REPORTS_DIR, PROFILING_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        return cls(
            hardware=hw,
            models=models,
            benchmarks=benchmarks,
            db_path=DB_PATH,
            experiments_dir=EXPERIMENTS_DIR,
            reports_dir=REPORTS_DIR,
            profiling_dir=PROFILING_DIR,
        )

    def summary(self) -> str:
        lines = [
            f"GPU: {self.hardware.gpu_name}",
            f"VRAM: {self.hardware.gpu_vram_mb} MB",
            f"CUDA: {self.hardware.cuda_version}",
            f"SMs: {self.hardware.sm_count}",
            f"Compute: sm_{self.hardware.compute_capability[0]}{self.hardware.compute_capability[1]}",
            f"DB: {self.db_path}",
        ]
        return "\n".join(lines)


# Singleton — lazy-initialized on first access
_config: Optional[LabConfig] = None


def get_config() -> LabConfig:
    global _config
    if _config is None:
        _config = LabConfig.build()
    return _config


if __name__ == "__main__":
    cfg = get_config()
    print(cfg.summary())
