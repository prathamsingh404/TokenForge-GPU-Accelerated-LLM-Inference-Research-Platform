"""
Deadline-Aware scheduler.

Each request has an SLA deadline (milliseconds from arrival).
The scheduler prioritizes requests closest to their deadline,
ensuring time-sensitive requests are completed on time.

Use cases:
- Real-time API endpoints with latency SLAs
- Interactive coding assistants (must respond within 200ms)
- Tiered service (premium users get tighter deadlines)
"""

import time

from tokenforge.schedulers.base import (
    BaseScheduler, ScheduledBatch, SchedulerRequest, RequestStatus,
)


class DeadlineAwareScheduler(BaseScheduler):
    """
    Earliest Deadline First (EDF) scheduler.

    Requests with the nearest deadline are scheduled first.
    Overdue requests get highest priority. Requests without
    deadlines are treated as lowest priority.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_in_batch: int = 4096,
        default_deadline_ms: float = 5000.0,
    ):
        super().__init__(max_batch_size, max_tokens_in_batch)
        self.default_deadline_ms = default_deadline_ms

    def _deadline_key(self, req: SchedulerRequest) -> float:
        """
        Sort key: lower = higher scheduling priority.

        Overdue requests: negative values (scheduled first)
        Near-deadline requests: small positive values
        No deadline: infinity (scheduled last)
        """
        remaining = req.deadline_remaining_ms
        if remaining is None:
            # Use default deadline
            elapsed_ms = (time.monotonic() - req.arrival_time) * 1000
            return self.default_deadline_ms - elapsed_ms
        return remaining

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()

        # Evict completed
        self._evict_completed()

        # Combine active + waiting, sort by deadline urgency
        all_candidates = list(self._active.values()) + list(self._waiting)
        all_candidates.sort(key=lambda r: self._deadline_key(r))

        # Rebuild
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

    def get_overdue_count(self) -> int:
        """Count how many active requests have missed their deadline."""
        return sum(
            1 for req in self._active.values() if req.is_overdue
        )

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["overdue_count"] = self.get_overdue_count()

        # Deadline achievement rate
        completed_with_deadline = [
            r for r in self._completed
            if r.deadline_ms is not None and r.total_latency is not None
        ]
        if completed_with_deadline:
            met = sum(
                1 for r in completed_with_deadline
                if (r.total_latency * 1000) <= r.deadline_ms
            )
            stats["deadline_achievement_rate"] = met / len(completed_with_deadline)
        else:
            stats["deadline_achievement_rate"] = 1.0

        return stats
