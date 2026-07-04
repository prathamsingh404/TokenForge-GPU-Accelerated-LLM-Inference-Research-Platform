"""
Token Fair scheduler.

Ensures fair distribution of GPU tokens across concurrent requests.
Tracks cumulative token allocations and prioritizes requests that
have received the fewest tokens relative to their peers.

Inspired by fair-share CPU schedulers (CFS in Linux). Prevents
any single request from monopolizing GPU resources.

Trade-off: Slightly lower throughput than FIFO, but guarantees
fairness and prevents starvation.
"""

from tokenforge.schedulers.base import (
    BaseScheduler, ScheduledBatch, SchedulerRequest, RequestStatus,
)


class TokenFairScheduler(BaseScheduler):
    """
    Token-fair scheduler — equitable GPU resource allocation.

    Each request tracks its cumulative token allocation. At each
    scheduling step, requests with the fewest allocated tokens
    are prioritized. This ensures no request is starved.

    Analogous to Completely Fair Scheduler (CFS) in Linux,
    but for GPU inference tokens instead of CPU time slices.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_tokens_in_batch: int = 4096,
    ):
        super().__init__(max_batch_size, max_tokens_in_batch)
        self._min_vruntime: float = 0.0  # Minimum virtual runtime

    def _vruntime(self, req: SchedulerRequest) -> float:
        """
        Virtual runtime: tokens allocated, adjusted by priority weight.

        Higher priority requests accumulate vruntime slower,
        so they get more tokens overall.
        """
        weight = max(1, req.priority + 1)  # priority 0 → weight 1, etc.
        return req.tokens_allocated / weight

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()

        # Evict completed
        self._evict_completed()

        # Set new requests' vruntime to the current minimum
        # so they don't get unfair advantage over existing requests
        for req in self._waiting:
            if req.tokens_allocated == 0:
                req.tokens_allocated = int(self._min_vruntime)

        # Combine all candidates, sort by vruntime (fewest tokens first)
        all_candidates = list(self._active.values()) + list(self._waiting)
        all_candidates.sort(key=lambda r: self._vruntime(r))

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

        # Update minimum vruntime
        if self._active:
            self._min_vruntime = min(
                self._vruntime(r) for r in self._active.values()
            )

        return batch

    def get_stats(self) -> dict:
        stats = super().get_stats()

        # Fairness metric: coefficient of variation of tokens allocated
        if self._completed:
            allocations = [r.tokens_allocated for r in self._completed]
            mean_alloc = sum(allocations) / len(allocations)
            if mean_alloc > 0:
                variance = sum((a - mean_alloc) ** 2 for a in allocations) / len(allocations)
                cv = (variance ** 0.5) / mean_alloc
                stats["fairness_cv"] = cv  # Lower = more fair
            else:
                stats["fairness_cv"] = 0.0

        return stats
