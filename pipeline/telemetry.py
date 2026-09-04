"""
GeoRecon AI - Hardware Telemetry Module (Phase 6.4)
Real-time thread-safe monitoring for CPU, RAM, GPU, VRAM, and Temperature.
Integrated with psutil and NVIDIA NVML (pynvml).
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Callable, Dict, Any, Optional, List
import warnings

import psutil

logger = logging.getLogger("GeoRecon.Telemetry")

# Safe NVML Import
NVML_AVAILABLE = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        import pynvml
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
except Exception as e:
    logger.warning("NVIDIA NVML could not be initialized: %s", e)
    pynvml = None
    NVML_AVAILABLE = False


@dataclass
class HardwareSnapshot:
    """Immutable real-time telemetry snapshot."""
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_name: str = "NVIDIA GeForce RTX 4060 Laptop GPU"
    gpu_util_percent: float = 0.0
    gpu_vram_used_mb: float = 0.0
    gpu_vram_total_mb: float = 0.0
    gpu_vram_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    nvml_available: bool = False
    timestamp: float = field(default_factory=time.time)
    # 60-Second Rolling History Buffers
    cpu_history: List[float] = field(default_factory=list)
    gpu_history: List[float] = field(default_factory=list)
    ram_history: List[float] = field(default_factory=list)


class HardwareTelemetryCollector:
    """Thread-safe continuous hardware telemetry collector running in a daemon thread."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks and Queues
        self._listeners: list[Callable[[HardwareSnapshot], None]] = []
        self._telemetry_queue: Optional[queue.Queue] = None

        # 60-Second Rolling History Buffers (120 points at 0.5s interval)
        self._history_len = int(60.0 / max(0.1, self.interval))
        self.cpu_history: deque = deque(maxlen=self._history_len)
        self.gpu_history: deque = deque(maxlen=self._history_len)
        self.ram_history: deque = deque(maxlen=self._history_len)

        # GPU Handle
        self._nvml_initialized = NVML_AVAILABLE
        self._gpu_handle = None
        self._gpu_name = "NVIDIA CUDA GPU"
        self._gpu_total_vram_mb = 8188.0

        if self._nvml_initialized and pynvml:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                raw_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                self._gpu_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self._gpu_total_vram_mb = round(mem.total / (1024 ** 2), 1)
            except Exception as e:
                logger.warning("Error fetching NVML device details: %s", e)

        # Fallback GPU detection if NVML name is generic
        if "CUDA" in self._gpu_name and shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if res.returncode == 0 and res.stdout.strip():
                    self._gpu_name = res.stdout.strip().splitlines()[0]
            except Exception:
                pass

        # Running Peak Trackers (for diagnostics.json)
        self.cpu_peak_percent: float = 0.0
        self.gpu_peak_percent: float = 0.0
        self.ram_peak_percent: float = 0.0
        self.gpu_vram_peak_mb: float = 0.0
        self.gpu_temperature_peak_c: float = 0.0

        # Prime psutil non-blocking CPU check
        psutil.cpu_percent(interval=None)

        # Latest Snapshot cache
        self._latest_snapshot: HardwareSnapshot = self._sample()

    def set_queue(self, q: queue.Queue) -> None:
        """Configures thread-safe queue for telemetry delivery to UI."""
        self._telemetry_queue = q

    def add_listener(self, callback: Callable[[HardwareSnapshot], None]) -> None:
        """Registers a subscriber callback."""
        with self._lock:
            self._listeners.append(callback)

    def reset_peaks(self) -> None:
        """Resets peak telemetry counters at the start of a reconstruction session."""
        with self._lock:
            self.cpu_peak_percent = 0.0
            self.gpu_peak_percent = 0.0
            self.ram_peak_percent = 0.0
            self.gpu_vram_peak_mb = 0.0
            self.gpu_temperature_peak_c = 0.0
            logger.info("Reset session peak telemetry counters.")

    def get_peak_metrics(self) -> Dict[str, Any]:
        """Returns peak hardware telemetry metrics for diagnostics.json."""
        with self._lock:
            return {
                "cpu_peak_percent": round(self.cpu_peak_percent, 1),
                "gpu_peak_percent": round(self.gpu_peak_percent, 1),
                "ram_peak_percent": round(self.ram_peak_percent, 1),
                "gpu_vram_peak_mb": round(self.gpu_vram_peak_mb, 1),
                "gpu_temperature_peak_c": round(self.gpu_temperature_peak_c, 1),
                # Title Case variants for flexible querying
                "CPU peak percent": round(self.cpu_peak_percent, 1),
                "GPU peak percent": round(self.gpu_peak_percent, 1),
                "RAM peak percent": round(self.ram_peak_percent, 1),
                "GPU VRAM peak MB": round(self.gpu_vram_peak_mb, 1),
                "GPU temperature peak C": round(self.gpu_temperature_peak_c, 1),
            }

    def _sample(self) -> HardwareSnapshot:
        """Polls current system resources and updates peaks atomically."""
        # 1. CPU & RAM (psutil)
        cpu_p = psutil.cpu_percent(interval=None)
        vmem = psutil.virtual_memory()
        ram_used = round(vmem.used / (1024 ** 3), 2)
        ram_tot = round(vmem.total / (1024 ** 3), 2)
        ram_p = vmem.percent

        # 2. GPU & VRAM & Temp (pynvml)
        gpu_util = 0.0
        vram_used = 0.0
        vram_tot = self._gpu_total_vram_mb
        vram_p = 0.0
        gpu_temp = 0.0

        if self._nvml_initialized and self._gpu_handle and pynvml:
            try:
                gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_used = round(mem.used / (1024 ** 2), 1)
                vram_tot = round(mem.total / (1024 ** 2), 1)
                vram_p = round((vram_used / max(1.0, vram_tot)) * 100.0, 1)
                gpu_temp = float(pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        else:
            # Safe Fallback: Estimate from PyTorch if loaded or nvidia-smi
            try:
                import torch
                if torch.cuda.is_available():
                    vram_used = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
                    vram_p = round((vram_used / max(1.0, vram_tot)) * 100.0, 1)
            except Exception:
                pass

        # 3. Update peaks and rolling history buffers atomically
        with self._lock:
            if cpu_p > self.cpu_peak_percent:
                self.cpu_peak_percent = cpu_p
            if gpu_util > self.gpu_peak_percent:
                self.gpu_peak_percent = gpu_util
            if ram_p > self.ram_peak_percent:
                self.ram_peak_percent = ram_p
            if vram_used > self.gpu_vram_peak_mb:
                self.gpu_vram_peak_mb = vram_used
            if gpu_temp > self.gpu_temperature_peak_c:
                self.gpu_temperature_peak_c = gpu_temp

            self.cpu_history.append(cpu_p)
            self.gpu_history.append(gpu_util)
            self.ram_history.append(ram_p)
            cpu_hist_copy = list(self.cpu_history)
            gpu_hist_copy = list(self.gpu_history)
            ram_hist_copy = list(self.ram_history)

        return HardwareSnapshot(
            cpu_percent=cpu_p,
            ram_used_gb=ram_used,
            ram_total_gb=ram_tot,
            ram_percent=ram_p,
            gpu_name=self._gpu_name,
            gpu_util_percent=gpu_util,
            gpu_vram_used_mb=vram_used,
            gpu_vram_total_mb=vram_tot,
            gpu_vram_percent=vram_p,
            gpu_temperature_c=gpu_temp,
            nvml_available=self._nvml_initialized,
            timestamp=time.time(),
            cpu_history=cpu_hist_copy,
            gpu_history=gpu_hist_copy,
            ram_history=ram_hist_copy,
        )

    def get_rolling_history(self) -> Dict[str, List[float]]:
        """Returns copies of the 60-second rolling history buffers."""
        with self._lock:
            return {
                "cpu": list(self.cpu_history),
                "gpu": list(self.gpu_history),
                "ram": list(self.ram_history),
            }

    def sample_now(self) -> HardwareSnapshot:
        """Synchronously captures and returns a telemetry snapshot."""
        snap = self._sample()
        self._latest_snapshot = snap
        return snap

    def get_latest_snapshot(self) -> HardwareSnapshot:
        """Returns the most recent cached snapshot."""
        return self._latest_snapshot

    def start(self) -> None:
        """Starts the background telemetry polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="GeoRecon-TelemetryCollector",
            daemon=True,
        )
        self._thread.start()
        logger.info("Hardware telemetry collection daemon started (interval=%ss).", self.interval)

    def stop(self) -> None:
        """Stops the background telemetry thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("Hardware telemetry collection daemon stopped.")

    def _worker_loop(self) -> None:
        """Continuous non-blocking collection loop."""
        while not self._stop_event.is_set():
            try:
                snapshot = self._sample()
                self._latest_snapshot = snapshot

                # Push to queue if configured (drops oldest if full to avoid lag)
                if self._telemetry_queue:
                    try:
                        self._telemetry_queue.put_nowait(snapshot)
                    except queue.Full:
                        try:
                            self._telemetry_queue.get_nowait()
                            self._telemetry_queue.put_nowait(snapshot)
                        except Exception:
                            pass

                # Dispatch to listeners
                with self._lock:
                    listeners = list(self._listeners)
                for listener in listeners:
                    try:
                        listener(snapshot)
                    except Exception as e:
                        logger.debug("Telemetry listener callback error: %s", e)

            except Exception as e:
                logger.debug("Telemetry sampling error: %s", e)

            self._stop_event.wait(self.interval)
