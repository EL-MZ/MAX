"""Sample Linux process RSS, JAX GPU allocator counters, and device-wide VRAM."""

import os
import shutil
import subprocess
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

MIB = 1024**2
METRICS = ("cpu_rss", "gpu_live", "gpu_pool", "gpu_device_used")


def read_cpu_memory(path=Path("/proc/self/status")):
    try:
        fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
    except OSError:
        return {}
    return {
        name: int(fields[key].split()[0]) / 1024
        for key, name in (("VmRSS", "cpu_rss_mib"), ("VmHWM", "cpu_process_hwm_mib"))
        if key in fields
    }


def read_jax_memory(device):
    if device is None:
        return {}
    try:
        stats = device.memory_stats() or {}
    except (RuntimeError, NotImplementedError):
        return {}
    names = {
        "bytes_in_use": "gpu_live_mib",
        "pool_bytes": "gpu_pool_mib",
        "peak_bytes_in_use": "gpu_allocator_lifetime_peak_mib",
        "peak_pool_bytes": "gpu_pool_lifetime_peak_mib",
        "bytes_limit": "gpu_allocator_limit_mib",
    }
    return {name: float(stats[key]) / MIB for key, name in names.items() if key in stats}


class NvidiaMemory:
    """Read whole-device usage, including other processes; not process VRAM."""

    def __init__(self, device):
        self.executable = shutil.which("nvidia-smi")
        if not self.executable and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
            self.executable = "/usr/lib/wsl/lib/nvidia-smi"
        ordinal = int(getattr(device, "local_hardware_id", device.id))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            self.selector = str(ordinal)
        else:
            selectors = [value.strip() for value in visible.split(",")]
            self.selector = selectors[ordinal] if ordinal < len(selectors) else ""
        self.available = bool(self.executable and self.selector)

    def __call__(self):
        if not self.available:
            return {}
        try:
            result = subprocess.run(
                [
                    self.executable,
                    f"--id={self.selector}",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            used, total = (float(value.strip()) for value in result.stdout.strip().split(","))
        except (OSError, ValueError, subprocess.SubprocessError):
            # Unsupported WSL/driver queries stay unavailable, never become zero.
            self.available = False
            return {}
        return {"gpu_device_used_mib": used, "gpu_device_total_mib": total}


class MemoryMonitor:
    """Per-pass sampled peaks; cumulative native high-water marks stay separate."""

    def __init__(
        self,
        device=None,
        interval=0.1,
        cpu_reader=read_cpu_memory,
        gpu_reader=None,
        system_reader=None,
    ):
        self.interval = interval
        self.cpu_reader = cpu_reader
        self.gpu_reader = gpu_reader or (lambda: read_jax_memory(device))
        self.system_reader = system_reader
        if system_reader is None and device is not None:
            self.system_reader = NvidiaMemory(device)
        self.stop_event, self.lock = Event(), Lock()
        self.thread = None
        self.last_system_poll = float("-inf")
        self.current, self.baseline, self.peaks = {}, {}, {}
        self.sample_count, self.system_sample_count = 0, 0

    def sample(self, force_system=False):
        values = {**self.cpu_reader(), **self.gpu_reader()}
        now = monotonic()
        system_polled = False
        if self.system_reader is not None and (
            force_system or now - self.last_system_poll >= max(1.0, self.interval)
        ):
            values.update(self.system_reader())
            self.last_system_poll = now
            system_polled = "gpu_device_used_mib" in values
        with self.lock:
            self.current.update(values)
            for metric in METRICS:
                value = values.get(f"{metric}_mib")
                if value is not None:
                    self.baseline.setdefault(metric, value)
                    self.peaks[metric] = max(value, self.peaks.get(metric, value))
            self.sample_count += 1
            self.system_sample_count += int(system_polled)

    def start(self):
        self.sample(force_system=True)
        self.thread = Thread(target=self._poll, name="tc-memory", daemon=True)
        self.thread.start()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exception):
        self.final_report = self.stop()

    def _poll(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample(force_system=True)
        return self.report()

    def report(self):
        with self.lock:
            result = {
                "memory_samples": self.sample_count,
                "gpu_device_memory_samples": self.system_sample_count,
                **{
                    name: self.current.get(name)
                    for name in (
                        "cpu_process_hwm_mib",
                        "gpu_allocator_lifetime_peak_mib",
                        "gpu_pool_lifetime_peak_mib",
                        "gpu_allocator_limit_mib",
                        "gpu_device_total_mib",
                    )
                },
            }
            for metric in METRICS:
                baseline, peak = self.baseline.get(metric), self.peaks.get(metric)
                result[f"{metric}_baseline_mib"] = baseline
                result[f"{metric}_peak_mib"] = peak
                result[f"{metric}_end_mib"] = self.current.get(f"{metric}_mib")
                result[f"{metric}_peak_increase_mib"] = (
                    max(0.0, peak - baseline) if baseline is not None and peak is not None else None
                )
            return result

    def progress(self):
        with self.lock:
            parts = []
            for metric, label in (
                ("cpu_rss", "RAM"),
                ("gpu_live", "GPU live"),
                ("gpu_pool", "GPU pool"),
                ("gpu_device_used", "GPU device"),
            ):
                current = self.current.get(f"{metric}_mib")
                peak = self.peaks.get(metric)
                if current is not None:
                    parts.append(f"{label} {current:.1f} MiB (peak {peak:.1f})")
            return " | ".join(parts) or "Memory N/A"


def combine_pass_memory(passes):
    """Case baseline is first pass; peaks span first pass and every warm repeat."""
    result = {}
    for metric in METRICS:
        baseline = passes[0][f"{metric}_baseline_mib"]
        peaks = [
            item[f"{metric}_peak_mib"] for item in passes if item[f"{metric}_peak_mib"] is not None
        ]
        peak = max(peaks) if peaks else None
        result[f"{metric}_baseline_mib"] = baseline
        result[f"{metric}_peak_mib"] = peak
        result[f"{metric}_end_mib"] = passes[-1][f"{metric}_end_mib"]
        result[f"{metric}_peak_increase_mib"] = (
            max(0.0, peak - baseline) if baseline is not None and peak is not None else None
        )
    for key in (
        "cpu_process_hwm_mib",
        "gpu_allocator_lifetime_peak_mib",
        "gpu_pool_lifetime_peak_mib",
        "gpu_allocator_limit_mib",
        "gpu_device_total_mib",
    ):
        result[key] = passes[-1][key]
    for key in ("memory_samples", "gpu_device_memory_samples"):
        result[key] = sum(item[key] for item in passes)
    return result


def format_mib(value):
    return "N/A" if value is None else f"{value:.1f}"
