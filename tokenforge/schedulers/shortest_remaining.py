"""
Shortest Remaining Processing Time (SRPT) scheduler.

Prioritizes requests that are closest to completion. This minimizes
average completion time — a well-known result in scheduling theory.

Trade-off: Can starve long requests. Combine with aging to prevent
permanent starvation.
"""

import time

from tokenforge.schedulers.base import (
    BaseScheduler, ScheduledBatch, SchedulerRequest, RequestStatus,
)


class ShortestRemainingScheduler(BaseScheduler):
    """
    SRPT scheduler — shortest remaining tokens first.

    At each scheduling step, the requests with the fewest remaining
    tokens to generate are prioritized. This minimizes the average
    request completion time.

    Optional aging: requests that have waited too long get a priority
    boost to prevent starvation.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_in_batch: int = 4096,
        aging_threshold_s: float = 30.0,
        aging_boost: int = 100,
    ):
        super().__init__(max_batch_size, max_tokens_in_batch)
        self.aging_threshold_s = aging_threshold_s
        self.aging_boost = aging_boost

    def _effective_remaining(self, req: SchedulerRequest) -> float:
        """Remaining tokens with aging adjustment."""
        remaining = req.remaining_tokens

        # Aging: reduce effective remaining for old requests
        wait = time.monotonic() - req.arrival_time
        if wait > self.aging_threshold_s:
            aging_factor = (wait - self.aging_threshold_s) * self.aging_boost
            remaining = max(0, remaining - aging_factor)

        return remaining

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()

        # Evict completed
        self._evict_completed()

        # All candidates: active + waiting, sorted by effective remaining
        all_candidates = list(self._active.values()) + list(self._waiting)
        all_candidates.sort(key=lambda r: self._effective_remaining(r))

        # Rebuild active set
        self._active.clear()
        new_waiting = []

        for i, req in enumerate(all_candidates):
            if i < self.max_batch_size:
                if req.generated_tokens > 0:
                    batch.decode_requests.append(req)
                else:
                    req.status = RequestStatus.PREFILLING
                    batch.prefill_requests.append(req)
                self._active[req.request_id] = req
            else:
                req.status = RequestStatus.PENDING
                new_waiting.append(req)

        self._waiting = new_waiting
        return batch
