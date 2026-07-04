"""
Environment manifest for benchmark reproducibility.

Every benchmark run automatically captures a complete snapshot
of the hardware, software, and configuration state. This makes
results fully reproducible and comparable across machines and
time.

Usage:
    from tokenforge.environment import EnvironmentManifest
    manifest = EnvironmentManifest.capture(model_name="gpt2")
    manifest.save("experiment_env.json")
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class EnvironmentManifest:
    """Complete environment snapshot for reproducibility."""

    # Hardware
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    gpu_driver_version: str = ""
    gpu_compute_capability: str = ""
    gpu_sm_count: int = 0

    # Software
    cuda_version: str = ""
    cudnn_version: str = ""
    pytorch_version: str = ""
    transformers_version: str = ""
    python_version: str = ""
    os_info: str = ""

    # Model
    model_name: str = ""
    quantization: str = ""

    # Project
    tokenforge_version: str = ""
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    git_is_dirty: bool = False

    # Timing
    timestamp: str = ""
    timestamp_unix: float = 0.0

    # Configuration snapshot
    config: dict = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        model_name: str = "",
        quantization: str = "",
        extra_config: Optional[dict] = None,
    ) -> "EnvironmentManifest":
        """
        Capture the current environment state.

        This is called automatically by the benchmark runner. Can also
        be called manually for debugging or environment auditing.
        """
        manifest = cls()

        # ── Hardware ──
        manifest._detect_gpu()

        # ── Software ──
        manifest._detect_software()

        # ── Model ──
        manifest.model_name = model_name
        manifest.quantization = quantization

        # ── Project ──
        manifest._detect_project_state()

        # ── Timing ──
        now = datetime.now(timezone.utc)
        manifest.timestamp = now.isoformat()
        manifest.timestamp_unix = now.timestamp()

        # ── Config ──
        if extra_config:
            manifest.config = extra_config

        return manifest

    def _detect_gpu(self):
        """Detect GPU hardware details via PyTorch + pynvml."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.gpu_name = props.name
                self.gpu_vram_mb = props.total_memory // (1024 ** 2)
                self.gpu_compute_capability = f"sm_{props.major}{props.minor}"
                self.gpu_sm_count = props.multi_processor_count
                self.cuda_version = torch.version.cuda or ""
        except ImportError:
            pass

        # Driver version via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_driver_version = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except Exception:
            self.gpu_driver_version = "unknown"

    def _detect_software(self):
        """Detect installed software versions."""
        self.python_version = platform.python_version()
        self.os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

        try:
            import torch
            self.pytorch_version = torch.__version__
        except ImportError:
            self.pytorch_version = "not installed"

        try:
            import transformers
            self.transformers_version = transformers.__version__
        except ImportError:
            self.transformers_version = "not installed"

        try:
            import torch.backends.cudnn as cudnn
            if cudnn.is_available():
                self.cudnn_version = str(cudnn.version())
        except Exception:
            self.cudnn_version = "unknown"

    def _detect_project_state(self):
        """Detect TokenForge version and git state."""
        try:
            from tokenforge import __version__
            self.tokenforge_version = __version__
        except ImportError:
            self.tokenforge_version = "unknown"

        # Git commit
        project_root = Path(__file__).resolve().parent.parent
        self.git_commit_hash = self._run_git(
            ["git", "rev-parse", "--short", "HEAD"], project_root,
        )
        self.git_branch = self._run_git(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], project_root,
        )

        dirty_check = self._run_git(
            ["git", "status", "--porcelain"], project_root,
        )
        self.git_is_dirty = bool(dirty_check and dirty_check.strip())

    @staticmethod
    def _run_git(cmd: list[str], cwd: Path) -> Optional[str]:
        """Run a git command and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=5, cwd=str(cwd),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return None

    def short_summary(self) -> str:
        """One-line summary for terminal output."""
        parts = [
            self.gpu_name,
            f"CUDA {self.cuda_version}",
            f"PyTorch {self.pytorch_version}",
        ]
        if self.git_commit_hash:
            parts.append(f"git:{self.git_commit_hash}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save(self, path: str | Path):
        """Save manifest as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "EnvironmentManifest":
        """Load manifest from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def matches(self, other: "EnvironmentManifest") -> dict:
        """
        Compare two manifests and return differences.

        Useful for checking if two experiments ran under
        identical conditions.
        """
        diffs = {}
        for field_name in [
            "gpu_name", "cuda_version", "pytorch_version",
            "transformers_version", "python_version",
            "model_name", "quantization",
        ]:
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            if a != b:
                diffs[field_name] = {"self": a, "other": b}
        return diffs
