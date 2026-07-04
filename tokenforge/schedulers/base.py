"""
Base scheduler interface.

All scheduling algorithms implement this ABC. The engine calls
`schedule()` every decode step to get the next batch of requests
to process. This decouples scheduling policy from the inference
engine, enabling drop-in comparison of different strategies.

To implement a custom scheduler:
    class MyScheduler(BaseScheduler):
        def schedule(self, waiting, active, max_batch) -> ScheduledBatch:
            # Your scheduling logic here
            ...
"""

import time
from abc import ABC, abstractmethod
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
class SchedulerRequest:
    """
    Unified request representation for the scheduler framework.

    Wraps the essential fields from InferenceRequest and SimulatedRequest
    into a single type that all schedulers operate on.
    """
    request_id: str
    prompt: str
    max_new_tokens: int = 64
    status: RequestStatus = RequestStatus.PENDING
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    generated_tokens: int = 0
    output_text: str = ""

    # For priority scheduling
    priority: int = 0  # Higher = more urgent

    # For deadline-aware scheduling
    deadline_ms: Optional[float] = None

    # For token-fair scheduling
    tokens_allocated: int = 0

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

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - self.generated_tokens)

    @property
    def wait_time(self) -> float:
        """Time spent waiting since arrival."""
        return time.monotonic() - self.arrival_time

    @property
    def deadline_remaining_ms(self) -> Optional[float]:
        """Milliseconds remaining until deadline."""
        if self.deadline_ms is None:
            return None
        elapsed_ms = (time.monotonic() - self.arrival_time) * 1000
        return self.deadline_ms - elapsed_ms

    @property
    def is_overdue(self) -> bool:
        remaining = self.deadline_remaining_ms
        return remaining is not None and remaining <= 0


@dataclass
class ScheduledBatch:
    """A batch of requests selected for one decode iteration."""
    prefill_requests: list[SchedulerRequest] = field(default_factory=list)
    decode_requests: list[SchedulerRequest] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def is_empty(self) -> bool:
        return self.total_size == 0

    @property
    def all_requests(self) -> list[SchedulerRequest]:
        return self.prefill_requests + self.decode_requests


class BaseScheduler(ABC):
    """
    Abstract base class for all scheduling algorithms.

    The inference engine calls schedule() at each iteration step.
    The scheduler decides which requests to process, in what order,
    and how to balance between new (prefill) and ongoing (decode)
    requests.

    Subclass this to implement custom scheduling strategies.
    """

    def __init__(self, max_batch_size: int = 8, max_tokens_in_batch: int = 4096):
        self.max_batch_size = max_batch_size
        self.max_tokens_in_batch = max_tokens_in_batch
        self._active: dict[str, SchedulerRequest] = {}
        self._waiting: list[SchedulerRequest] = []
        self._completed: list[SchedulerRequest] = []

    def add_requests(self, requests: list[SchedulerRequest]):
        """Add new requests to the waiting queue."""
        for req in requests:
            req.status = RequestStatus.PENDING
            self._waiting.append(req)

    @abstractmethod
    def schedule(self) -> ScheduledBatch:
        """
        Build the next batch for one decode step.

        Must be implemented by each scheduler. Should:
        1. Continue decoding active requests
        2. Fill remaining slots according to the scheduling policy
        3. Return a ScheduledBatch
        """
        ...

    def mark_token_generated(self, request_id: str):
        """Notify scheduler that a token was generated for a request."""
        if request_id in self._active:
            req = self._active[request_id]
            if req.generated_tokens == 0:
                req.first_token_time = time.monotonic()
            req.generated_tokens += 1
            req.tokens_allocated += 1
            req.status = RequestStatus.DECODING

    def mark_completed(self, request_id: str):
        """Mark a request as finished."""
        if request_id in self._active:
            req = self._active.pop(request_id)
            req.status = RequestStatus.COMPLETED
            req.completion_time = time.monotonic()
            self._completed.append(req)

    def _evict_completed(self) -> list[str]:
        """Remove completed requests from active set. Returns evicted IDs."""
        completed_ids = []
        for rid, req in list(self._active.items()):
            if req.status == RequestStatus.COMPLETED:
                completed_ids.append(rid)
            elif req.generated_tokens >= req.max_new_tokens:
                req.status = RequestStatus.COMPLETED
                req.completion_time = time.monotonic()
                completed_ids.append(rid)

        for rid in completed_ids:
            req = self._active.pop(rid)
            self._completed.append(req)

        return completed_ids

    @property
    def has_work(self) -> bool:
        return bool(self._active) or bool(self._waiting)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def completed_requests(self) -> list[SchedulerRequest]:
        return list(self._completed)

    def get_stats(self) -> dict:
        """Aggregate statistics for the scheduler's completed requests."""
        ttfts = [r.ttft for r in self._completed if r.ttft is not None]
        latencies = [r.total_latency for r in self._completed if r.total_latency is not None]
        waits = [r.wait_time for r in self._completed]
        tokens = [r.generated_tokens for r in self._completed]

        def safe_avg(vals):
            return sum(vals) / len(vals) if vals else 0

        return {
            "total_completed": len(self._completed),
            "active": self.active_count,
            "pending": self.waiting_count,
            "avg_ttft": safe_avg(ttfts),
            "avg_latency": safe_avg(latencies),
            "total_tokens": sum(tokens),
        }
