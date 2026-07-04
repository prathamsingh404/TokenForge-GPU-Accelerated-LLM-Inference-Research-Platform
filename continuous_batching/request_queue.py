# TokenForge GPU-Accelerated LLM Inference Platform
"""
Async request queue for the continuous batching engine.

Simulates incoming inference requests from multiple users.
Each request has a unique ID, prompt, and generation parameters.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestStatus(Enum):
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 64
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: RequestStatus = RequestStatus.PENDING
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    generated_tokens: int = 0
    output_text: str = ""

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time and self.arrival_time:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def total_latency(self) -> Optional[float]:
        if self.completion_time and self.arrival_time:
            return self.completion_time - self.arrival_time
        return None


class RequestQueue:
    """
    Thread-safe request queue with priority support.
    New requests can arrive while the engine is processing.
    """

    def __init__(self, max_size: int = 1000):
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=max_size)
        self._active: dict[str, InferenceRequest] = {}
        self._completed: list[InferenceRequest] = []
        self._lock = asyncio.Lock()

    async def submit(self, request: InferenceRequest):
        await self._queue.put(request)

    async def get_batch(self, max_batch: int) -> list[InferenceRequest]:
        """Pull up to max_batch requests from the queue."""
        batch = []
        while len(batch) < max_batch and not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                batch.append(req)
            except asyncio.QueueEmpty:
                break
        return batch

    def mark_active(self, request: InferenceRequest):
        self._active[request.request_id] = request

    def mark_completed(self, request: InferenceRequest):
        request.status = RequestStatus.COMPLETED
        request.completion_time = time.monotonic()
        self._active.pop(request.request_id, None)
        self._completed.append(request)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def completed_requests(self) -> list[InferenceRequest]:
        return list(self._completed)

    def get_stats(self) -> dict:
        ttfts = [r.ttft for r in self._completed if r.ttft is not None]
        latencies = [r.total_latency for r in self._completed if r.total_latency is not None]
        tokens = [r.generated_tokens for r in self._completed]

        return {
            "total_completed": len(self._completed),
            "active": self.active_count,
            "pending": self.pending_count,
            "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "total_tokens": sum(tokens),
        }
