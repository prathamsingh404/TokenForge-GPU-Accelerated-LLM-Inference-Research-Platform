# TokenForge GPU-Accelerated LLM Inference Platform
"""
Request scheduler for continuous batching.

Decides which requests enter the GPU batch at each iteration.
Implements FCFS with preemption support — finished requests
are immediately replaced by waiting ones.
"""

import time
from typing import Optional
from dataclasses import dataclass, field

from continuous_batching.request_queue import InferenceRequest, RequestStatus


@dataclass
class ScheduledBatch:
    """A batch of requests ready for one decode iteration."""
    prefill_requests: list[InferenceRequest] = field(default_factory=list)
    decode_requests: list[InferenceRequest] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def is_empty(self) -> bool:
        return self.total_size == 0


class ContinuousBatchScheduler:
    """
    Iteration-level scheduler inspired by Orca/vLLM.

    Key idea: instead of waiting to fill a batch and process it
    as a unit, we schedule at the granularity of individual decode
    steps. Finished requests leave immediately and new requests
    can join without waiting for the entire batch to complete.

    This keeps GPU utilization high even when requests have
    different lengths.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_in_batch: int = 4096,
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_in_batch = max_tokens_in_batch

        # Requests currently being decoded
        self._active: dict[str, InferenceRequest] = {}
        # Waiting requests (already submitted but not yet in a batch)
        self._waiting: list[InferenceRequest] = []

    def add_requests(self, requests: list[InferenceRequest]):
        """Add new requests to the waiting queue."""
        for req in requests:
            req.status = RequestStatus.PENDING
            self._waiting.append(req)

    def schedule(self) -> ScheduledBatch:
        """
        Build the next batch for one decode step.

        Priority:
        1. Continue decoding active requests (already in flight)
        2. Fill remaining slots with new prefill requests
        """
        batch = ScheduledBatch()

        # Carry forward active (decoding) requests
        completed_ids = []
        for rid, req in self._active.items():
            if req.status == RequestStatus.COMPLETED:
                completed_ids.append(rid)
            elif req.generated_tokens >= req.max_new_tokens:
                req.status = RequestStatus.COMPLETED
                req.completion_time = time.monotonic()
                completed_ids.append(rid)
            else:
                batch.decode_requests.append(req)

        for rid in completed_ids:
            self._active.pop(rid)

        # Fill remaining slots with waiting requests
        available_slots = self.max_batch_size - batch.total_size
        new_requests = []

        while available_slots > 0 and self._waiting:
            req = self._waiting.pop(0)
            req.status = RequestStatus.PREFILLING
            new_requests.append(req)
            self._active[req.request_id] = req
            available_slots -= 1

        batch.prefill_requests = new_requests
        return batch

    def mark_token_generated(self, request_id: str):
        """Notify scheduler that a token was generated for a request."""
        if request_id in self._active:
            req = self._active[request_id]
            if req.generated_tokens == 0:
                req.first_token_time = time.monotonic()
            req.generated_tokens += 1
            req.status = RequestStatus.DECODING

    def mark_completed(self, request_id: str):
        """Mark a request as finished."""
        if request_id in self._active:
            req = self._active[request_id]
            req.status = RequestStatus.COMPLETED
            req.completion_time = time.monotonic()

    @property
    def has_work(self) -> bool:
        return bool(self._active) or bool(self._waiting)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    def get_completed(self) -> list[InferenceRequest]:
        return [
            r for r in self._active.values()
            if r.status == RequestStatus.COMPLETED
        ]
