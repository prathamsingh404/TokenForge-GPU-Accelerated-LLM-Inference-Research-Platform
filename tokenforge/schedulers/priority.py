"""
Priority Queue scheduler.

Processes requests based on priority level (higher = more urgent).
Within the same priority level, FIFO ordering is used.

Use cases:
- VIP customers get faster service
- System prompts > user prompts
- Real-time requests > batch requests
"""

from tokenforge.schedulers.base import (
    BaseScheduler, ScheduledBatch, SchedulerRequest, RequestStatus,
)


class PriorityScheduler(BaseScheduler):
    """
    Priority-based scheduler.

    Requests with higher priority values are scheduled first.
    Within the same priority, arrival order is preserved (stable sort).
    Active requests always continue (no preemption) — priority only
    affects which waiting requests get admitted.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_in_batch: int = 4096,
        preemptive: bool = False,
    ):
        super().__init__(max_batch_size, max_tokens_in_batch)
        self.preemptive = preemptive

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()

        # Evict completed
        self._evict_completed()

        if self.preemptive:
            # Preemptive: re-evaluate all requests including active
            all_requests = list(self._active.values()) + list(self._waiting)
            all_requests.sort(
                key=lambda r: (-r.priority, r.arrival_time),
            )

            # Reset active set
            self._active.clear()
            new_waiting = []

            for i, req in enumerate(all_requests):
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
        else:
            # Non-preemptive: active requests continue, priority for new admissions
            for req in self._active.values():
                batch.decode_requests.append(req)

            # Sort waiting by priority (descending), then arrival (ascending)
            self._waiting.sort(
                key=lambda r: (-r.priority, r.arrival_time),
            )

            # Fill remaining slots
            available_slots = self.max_batch_size - batch.total_size
            admitted = []
            remaining = []

            for req in self._waiting:
                if available_slots > 0:
                    req.status = RequestStatus.PREFILLING
                    self._active[req.request_id] = req
                    batch.prefill_requests.append(req)
                    available_slots -= 1
                    admitted.append(req)
                else:
                    remaining.append(req)

            self._waiting = remaining

        return batch
