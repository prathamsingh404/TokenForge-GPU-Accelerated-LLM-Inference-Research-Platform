# TokenForge GPU-Accelerated LLM Inference Platform
"""
Real-time GPU metrics collection via pynvml.

Runs a background polling thread that samples GPU utilization, VRAM,
temperature, power draw, and clock speed. Designed to wrap benchmark
runs as a context manager.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import pynvml


@dataclass
class GPUSnapshot:
    gpu_util_percent: float
    vram_used_mb: float
    vram_total_mb: float
    temperature_c: float
    power_draw_w: float
    clock_mhz: float
    timestamp: float  # time.monotonic()


@dataclass
class GPUMetricsSummary:
    """Aggregated GPU stats over a benchmark run."""
    avg_util: float
    max_util: float
    avg_vram_mb: float
    peak_vram_mb: float
    avg_temperature: float
    max_temperature: float
    avg_power_w: float
    max_power_w: float
    sample_count: int
    energy_joules: float = 0.0  # Cumulative energy consumed (power × time)
    snapshots: list[GPUSnapshot] = field(default_factory=list)


class GPUMonitor:
    """
    Polls GPU metrics on a background thread. Use as a context manager
    around benchmark code to automatically capture GPU behavior.

    Usage:
        monitor = GPUMonitor(poll_interval_ms=100)
        with monitor:
            run_benchmark()
        summary = monitor.summarize()
    """

    def __init__(
        self,
        device_index: int = 0,
        poll_interval_ms: int = 100,
        keep_snapshots: bool = True,
    ):
        self.device_index = device_index
        self.poll_interval = poll_interval_ms / 1000.0
        self.keep_snapshots = keep_snapshots

        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._snapshots: list[GPUSnapshot] = []
        self._initialized = False

    def _init_nvml(self):
        if not self._initialized:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._initialized = True

    def _shutdown_nvml(self):
        if self._initialized:
            pynvml.nvmlShutdown()
            self._initialized = False

    def _sample(self) -> GPUSnapshot:
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            self._handle, pynvml.NVML_TEMPERATURE_GPU
        )
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW → W
        except pynvml.NVMLError:
            power = 0.0
        try:
            clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            clock = 0.0

        return GPUSnapshot(
            gpu_util_percent=float(util.gpu),
            vram_used_mb=mem_info.used / (1024 ** 2),
            vram_total_mb=mem_info.total / (1024 ** 2),
            temperature_c=float(temp),
            power_draw_w=power,
            clock_mhz=float(clock),
            timestamp=time.monotonic(),
        )

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                snap = self._sample()
                self._snapshots.append(snap)
            except pynvml.NVMLError:
                pass  # GPU momentarily busy, skip this sample
            self._stop_event.wait(self.poll_interval)

    def start(self):
        self._init_nvml()
        self._snapshots.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._shutdown_nvml()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def get_current(self) -> Optional[GPUSnapshot]:
        """Get the most recent snapshot, or take a fresh one."""
        self._init_nvml()
        try:
            return self._sample()
        except pynvml.NVMLError:
            return None
        finally:
            if not self._thread:
                self._shutdown_nvml()

    def summarize(self) -> GPUMetricsSummary:
        if not self._snapshots:
            return GPUMetricsSummary(
                avg_util=0, max_util=0,
                avg_vram_mb=0, peak_vram_mb=0,
                avg_temperature=0, max_temperature=0,
                avg_power_w=0, max_power_w=0,
                sample_count=0,
            )

        n = len(self._snapshots)
        utils = [s.gpu_util_percent for s in self._snapshots]
        vrams = [s.vram_used_mb for s in self._snapshots]
        temps = [s.temperature_c for s in self._snapshots]
        powers = [s.power_draw_w for s in self._snapshots]

        # Compute cumulative energy (Joules) via trapezoidal integration
        # of power over time between consecutive snapshots
        energy_joules = 0.0
        if n > 1:
            for i in range(1, n):
                dt = self._snapshots[i].timestamp - self._snapshots[i - 1].timestamp
                avg_power = (self._snapshots[i].power_draw_w
                             + self._snapshots[i - 1].power_draw_w) / 2.0
                energy_joules += avg_power * dt

        return GPUMetricsSummary(
            avg_util=sum(utils) / n,
            max_util=max(utils),
            avg_vram_mb=sum(vrams) / n,
            peak_vram_mb=max(vrams),
            avg_temperature=sum(temps) / n,
            max_temperature=max(temps),
            avg_power_w=sum(powers) / n,
            max_power_w=max(powers),
            sample_count=n,
            energy_joules=energy_joules,
            snapshots=self._snapshots if self.keep_snapshots else [],
        )


# Convenience function for one-shot readings
def read_gpu_stats(device_index: int = 0) -> Optional[GPUSnapshot]:
    monitor = GPUMonitor(device_index=device_index)
    return monitor.get_current()


if __name__ == "__main__":
    snap = read_gpu_stats()
    if snap:
        print(f"GPU Util:    {snap.gpu_util_percent:.0f}%")
        print(f"VRAM Used:   {snap.vram_used_mb:.0f} / {snap.vram_total_mb:.0f} MB")
        print(f"Temperature: {snap.temperature_c:.0f}°C")
        print(f"Power Draw:  {snap.power_draw_w:.1f} W")
        print(f"Clock:       {snap.clock_mhz:.0f} MHz")
    else:
        print("Could not read GPU metrics.")
