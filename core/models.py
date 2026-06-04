"""
Pydantic models for the dashboard REST API.

Shared between the FastAPI backend and any client that needs
typed request/response payloads.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


# --- Request Models ---

class RunBenchmarkRequest(BaseModel):
    phase: str = Field(..., description="Which phase to run: quantization, batching, etc.")
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    quantization: Optional[str] = None
    sequence_length: Optional[int] = None
    max_new_tokens: int = 128
    num_runs: int = 10


class ExperimentFilter(BaseModel):
    phase: Optional[str] = None
    model_name: Optional[str] = None
    limit: int = 100


# --- Response Models ---

class ExperimentResponse(BaseModel):
    id: str
    name: str
    phase: str
    model_name: str
    batch_size: Optional[int]
    quantization: Optional[str]
    sequence_length: Optional[int]
    created_at: str


class MetricResponse(BaseModel):
    metric_name: str
    metric_value: float
    unit: str
    recorded_at: str


class GPUSnapshotResponse(BaseModel):
    gpu_util_percent: float
    vram_used_mb: float
    vram_total_mb: float
    temperature_c: float
    power_draw_w: float
    clock_mhz: float = 0.0
    recorded_at: str


class ExperimentDetailResponse(BaseModel):
    experiment: ExperimentResponse
    metrics: list[MetricResponse]
    gpu_snapshots: list[GPUSnapshotResponse]


class ComparisonPoint(BaseModel):
    experiment_id: str
    experiment_name: str
    model_name: str
    batch_size: Optional[int]
    quantization: Optional[str]
    metric_value: float
    unit: str


class GPULiveResponse(BaseModel):
    gpu_util_percent: float
    vram_used_mb: float
    vram_total_mb: float
    temperature_c: float
    power_draw_w: float
    clock_mhz: float
    gpu_name: str


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu_available: bool
    gpu_name: str
    vram_total_mb: int
    experiments_count: int
    cuda_version: str
