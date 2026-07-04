"""
Round Robin scheduler.

Cycles through waiting requests, giving each a fixed number of
decode steps before moving to the next. This ensures all requests
make forward progress, preventing starvation of long-running
requests by short ones.

Trade-off: Lower throughput than FIFO but better fairness.
"""

from tokenforge.schedulers.base import (
    BaseScheduler, ScheduledBatch, SchedulerRequest, RequestStatus,
)


class RoundRobinScheduler(BaseScheduler):
    """
    Round Robin scheduler with configurable quantum.

    Each request gets `quantum` decode steps before being rotated
    to the back of the queue. Active requests are cycled fairly.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_in_batch: int = 4096,
        quantum: int = 4,
    ):
        super().__init__(max_batch_size, max_tokens_in_batch)
        self.quantum = quantum
        self._step_counts: dict[str, int] = {}  # steps since last rotation

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()

        # Evict completed
        evicted = self._evict_completed()
        for rid in evicted:
            self._step_counts.pop(rid, None)

        # Check if any active requests have exhausted their quantum
        rotate_ids = []
        for rid, req in list(self._active.items()):
            steps = self._step_counts.get(rid, 0)
            if steps >= self.quantum:
                rotate_ids.append(rid)

        # Rotate exhausted requests back to waiting
        for rid in rotate_ids:
            req = self._active.pop(rid)
            req.status = RequestStatus.PENDING
            self._waiting.append(req)
            self._step_counts.pop(rid, None)

        # Carry forward remaining active requests
        for req in self._active.values():
            batch.decode_requests.append(req)
            self._step_counts[req.request_id] = (
                self._step_counts.get(req.request_id, 0) + 1
            )

        # Fill remaining slots from waiting queue
        available_slots = self.max_batch_size - batch.total_size
        while available_slots > 0 and self._waiting:
            req = self._waiting.pop(0)
            req.status = RequestStatus.PREFILLING
            self._active[req.request_id] = req
            batch.prefill_requests.append(req)
            self._step_counts[req.request_id] = 1
            available_slots -= 1

        return batch
