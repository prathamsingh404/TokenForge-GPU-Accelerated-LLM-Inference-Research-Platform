"""
FIFO (First-In-First-Out) scheduler.

The simplest scheduling algorithm: requests are processed in
arrival order. This is equivalent to the continuous batching
scheduler already in the project, extracted as a plugin.

Use as baseline when comparing more advanced strategies.
"""

from tokenforge.schedulers.base import (
    BaseScheduler, ScheduledBatch, SchedulerRequest, RequestStatus,
)


class FIFOScheduler(BaseScheduler):
    """
    First-Come-First-Served scheduler.

    Priority:
    1. Continue decoding active requests (already in flight)
    2. Fill remaining slots with waiting requests in arrival order
    """

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()

        # Evict completed requests
        self._evict_completed()

        # Carry forward active requests
        for req in self._active.values():
            batch.decode_requests.append(req)

        # Fill remaining slots from waiting queue (FIFO order)
        available_slots = self.max_batch_size - batch.total_size
        while available_slots > 0 and self._waiting:
            req = self._waiting.pop(0)
            req.status = RequestStatus.PREFILLING
            self._active[req.request_id] = req
            batch.prefill_requests.append(req)
            available_slots -= 1

        return batch
