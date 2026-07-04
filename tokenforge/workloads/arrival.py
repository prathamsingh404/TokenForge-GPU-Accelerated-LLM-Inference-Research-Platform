"""
Arrival pattern generators for workload simulation.

Generates request arrival timestamps using various statistical
distributions that model real-world traffic patterns.

Supported patterns:
    - poisson:  Standard Poisson process (exponential inter-arrivals).
                Models: independent user requests arriving randomly.
    - uniform:  Constant inter-arrival time.
                Models: batch processing, periodic polling.
    - burst:    Alternating high/low activity phases.
                Models: traffic spikes (launch events, news breaks).
    - trace:    Replay arrival times from a recorded trace file.
                Models: exact reproduction of real traffic.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArrivalEvent:
    """A single arrival event in the workload timeline."""
    timestamp: float  # Seconds from simulation start
    request_index: int


def poisson_arrivals(
    rate: float,
    duration: float,
    seed: Optional[int] = None,
) -> list[ArrivalEvent]:
    """
    Generate Poisson process arrivals.

    Args:
        rate: Mean arrival rate (requests per second).
        duration: Total simulation duration in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of ArrivalEvent with timestamps.
    """
    rng = random.Random(seed)
    events = []
    t = 0.0
    idx = 0

    while t < duration:
        # Exponential inter-arrival time
        inter_arrival = -math.log(1.0 - rng.random()) / rate
        t += inter_arrival
        if t < duration:
            events.append(ArrivalEvent(timestamp=t, request_index=idx))
            idx += 1

    return events


def uniform_arrivals(
    rate: float,
    duration: float,
    jitter: float = 0.0,
    seed: Optional[int] = None,
) -> list[ArrivalEvent]:
    """
    Generate uniformly spaced arrivals with optional jitter.

    Args:
        rate: Requests per second.
        duration: Total simulation duration in seconds.
        jitter: Random jitter as fraction of inter-arrival time (0.0-1.0).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    interval = 1.0 / rate if rate > 0 else duration
    events = []
    t = 0.0
    idx = 0

    while t < duration:
        noise = rng.uniform(-jitter, jitter) * interval if jitter > 0 else 0.0
        actual_t = max(0.0, t + noise)
        if actual_t < duration:
            events.append(ArrivalEvent(timestamp=actual_t, request_index=idx))
            idx += 1
        t += interval

    events.sort(key=lambda e: e.timestamp)
    return events


def burst_arrivals(
    base_rate: float,
    burst_rate: float,
    duration: float,
    burst_duration: float = 5.0,
    burst_interval: float = 30.0,
    seed: Optional[int] = None,
) -> list[ArrivalEvent]:
    """
    Generate bursty arrivals that alternate between low and high activity.

    Args:
        base_rate: Low-activity arrival rate (req/s).
        burst_rate: High-activity arrival rate (req/s).
        duration: Total simulation duration in seconds.
        burst_duration: How long each burst lasts (seconds).
        burst_interval: Time between burst starts (seconds).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    events = []
    t = 0.0
    idx = 0

    while t < duration:
        # Determine if we're in a burst phase
        cycle_pos = t % burst_interval
        in_burst = cycle_pos < burst_duration
        rate = burst_rate if in_burst else base_rate

        inter_arrival = -math.log(1.0 - rng.random()) / rate
        t += inter_arrival
        if t < duration:
            events.append(ArrivalEvent(timestamp=t, request_index=idx))
            idx += 1

    return events


def trace_arrivals(
    timestamps: list[float],
    time_scale: float = 1.0,
) -> list[ArrivalEvent]:
    """
    Replay arrivals from a recorded trace.

    Args:
        timestamps: List of arrival times in seconds.
        time_scale: Scaling factor (< 1.0 = faster, > 1.0 = slower).
    """
    events = []
    for idx, ts in enumerate(sorted(timestamps)):
        events.append(ArrivalEvent(
            timestamp=ts * time_scale,
            request_index=idx,
        ))
    return events


# Registry of available arrival generators
ARRIVAL_GENERATORS = {
    "poisson": poisson_arrivals,
    "uniform": uniform_arrivals,
    "burst": burst_arrivals,
    "trace": trace_arrivals,
}
