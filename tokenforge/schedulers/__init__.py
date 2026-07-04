"""
TokenForge scheduler framework.

Pluggable scheduling algorithms for inference request management.
Users can compare FIFO, Priority, Round Robin, and other strategies.

Usage:
    from tokenforge.schedulers import get_scheduler, list_schedulers

    scheduler = get_scheduler("priority", max_batch_size=16)
"""

from tokenforge.schedulers.base import BaseScheduler, ScheduledBatch


_SCHEDULER_REGISTRY: dict[str, type] = {}


def register_scheduler(name: str, cls: type):
    """Register a scheduler implementation."""
    _SCHEDULER_REGISTRY[name] = cls


def get_scheduler(name: str, **kwargs) -> BaseScheduler:
    """Instantiate a scheduler by name."""
    if name not in _SCHEDULER_REGISTRY:
        available = ", ".join(_SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    return _SCHEDULER_REGISTRY[name](**kwargs)


def list_schedulers() -> list[str]:
    """Return names of all registered schedulers."""
    return list(_SCHEDULER_REGISTRY.keys())


# Auto-register built-in schedulers on import
def _auto_register():
    from tokenforge.schedulers.fifo import FIFOScheduler
    from tokenforge.schedulers.round_robin import RoundRobinScheduler
    from tokenforge.schedulers.priority import PriorityScheduler
    from tokenforge.schedulers.shortest_remaining import ShortestRemainingScheduler
    from tokenforge.schedulers.deadline_aware import DeadlineAwareScheduler
    from tokenforge.schedulers.token_fair import TokenFairScheduler

    register_scheduler("fifo", FIFOScheduler)
    register_scheduler("continuous", FIFOScheduler)  # Alias for backwards compat
    register_scheduler("round_robin", RoundRobinScheduler)
    register_scheduler("priority", PriorityScheduler)
    register_scheduler("shortest_remaining", ShortestRemainingScheduler)
    register_scheduler("deadline_aware", DeadlineAwareScheduler)
    register_scheduler("token_fair", TokenFairScheduler)


_auto_register()
