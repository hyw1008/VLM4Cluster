from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Iterator

from vlm4cluster.utils.deps import require_module


_MEBIBYTE = 1024 * 1024
_CURRENT_RECORDER: ContextVar[TrainingEfficiencyRecorder | None] = ContextVar(
    "vlm4cluster_training_efficiency_recorder",
    default=None,
)


@dataclass(slots=True)
class TrainingEfficiencyRecorder:
    enabled: bool = True
    run_device: Any = "cpu"
    train_eval_time_seconds: float = 0.0
    train_eval_phase_count: int = 0
    train_eval_device: str | None = None
    peak_cpu_memory_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    _active_measurements: list[TrainingMeasurement] = field(default_factory=list, repr=False)
    _run_memory_measurement: _RunPeakMemoryMeasurement | None = field(default=None, repr=False)
    _memory_suspension_depth: int = field(default=0, repr=False)

    def start_run_memory_measurement(self) -> None:
        if not self.enabled or self._run_memory_measurement is not None:
            return
        measurement = _RunPeakMemoryMeasurement(self.run_device)
        measurement.start()
        self._run_memory_measurement = measurement

    def stop_run_memory_measurement(self) -> None:
        if self._run_memory_measurement is None:
            return
        measurement = self._run_memory_measurement
        self._run_memory_measurement = None
        self.peak_cpu_memory_bytes, self.peak_gpu_memory_bytes = measurement.stop()

    def suspend_run_memory_measurement(self) -> None:
        if not self.enabled or self._run_memory_measurement is None:
            return
        if self._memory_suspension_depth == 0:
            self._run_memory_measurement.pause()
        self._memory_suspension_depth += 1

    def resume_run_memory_measurement(self) -> None:
        if not self.enabled or self._run_memory_measurement is None:
            return
        if self._memory_suspension_depth <= 0:
            raise RuntimeError("Efficiency memory measurement is not suspended.")
        self._memory_suspension_depth -= 1
        if self._memory_suspension_depth == 0:
            self._run_memory_measurement.resume()

    def register(self, measurement: TrainingMeasurement) -> None:
        self._active_measurements.append(measurement)

    def discard(self, measurement: TrainingMeasurement) -> None:
        if measurement in self._active_measurements:
            self._active_measurements.remove(measurement)

    def stop_active_measurements(self) -> None:
        for measurement in tuple(self._active_measurements):
            measurement.cancel()

    def record_train_eval_time(self, *, elapsed_seconds: float, device: str) -> None:
        self.train_eval_time_seconds += elapsed_seconds
        self.train_eval_phase_count += 1
        self.train_eval_device = device

    def as_dict(self, *, checkpoint_loaded: bool = False) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        train_eval_measured = self.train_eval_phase_count > 0
        if train_eval_measured:
            status = "measured"
        elif checkpoint_loaded:
            status = "checkpoint_reused"
        else:
            status = "not_run"
        return {
            "status": status,
            "dataset_scope": "imagenet_1k",
            "train_eval_measured": train_eval_measured,
            "train_eval_time_seconds": (
                round(self.train_eval_time_seconds, 6) if train_eval_measured else None
            ),
            "memory_measured": self.peak_cpu_memory_bytes is not None,
            "peak_cpu_memory_mb": (
                round(self.peak_cpu_memory_bytes / _MEBIBYTE, 3)
                if self.peak_cpu_memory_bytes is not None
                else None
            ),
            "peak_gpu_memory_mb": (
                round(self.peak_gpu_memory_bytes / _MEBIBYTE, 3)
                if self.peak_gpu_memory_bytes is not None
                else None
            ),
            "cpu_memory_metric": (
                "process_tree_rss" if self.peak_cpu_memory_bytes is not None else None
            ),
            "gpu_memory_metric": (
                "torch.cuda.max_memory_allocated"
                if self.peak_gpu_memory_bytes is not None
                else None
            ),
            "memory_scope": "dataset_preparation_through_metric_evaluation",
            "memory_excluded_phases": ["anyattack_image_generation"],
            "memory_unit": "MiB",
            "run_device": str(self.run_device),
            "train_eval_device": self.train_eval_device,
            "train_eval_phase_count": self.train_eval_phase_count,
        }


class TrainingMeasurement:
    def __init__(self, recorder: TrainingEfficiencyRecorder | None, device: Any) -> None:
        self._recorder = recorder
        self._device = device
        self._torch = None
        self._cuda_device = None
        self._start_time: float | None = None
        self._stopped = False
        if recorder is None or not recorder.enabled:
            return

        torch = require_module("torch", "pip install torch")
        resolved_device = torch.device(device)
        self._torch = torch
        if resolved_device.type == "cuda" and torch.cuda.is_available():
            self._cuda_device = resolved_device
            torch.cuda.synchronize(resolved_device)
        self._start_time = time.perf_counter()
        recorder.register(self)

    def stop(self) -> None:
        if self._stopped or self._recorder is None or self._start_time is None:
            return
        self._stopped = True
        try:
            if self._cuda_device is not None:
                self._torch.cuda.synchronize(self._cuda_device)
            elapsed_seconds = time.perf_counter() - self._start_time
            self._recorder.record_train_eval_time(
                elapsed_seconds=elapsed_seconds,
                device=str(self._device),
            )
        finally:
            self._recorder.discard(self)

    def cancel(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._recorder is not None:
            self._recorder.discard(self)


class _RunPeakMemoryMeasurement:
    def __init__(self, device: Any) -> None:
        torch = require_module("torch", "pip install torch")
        resolved_device = torch.device(device)
        self._torch = torch
        self._cuda_device = (
            resolved_device
            if resolved_device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._cpu_sampler = _PeakRSSSampler()
        self._peak_gpu_memory_bytes: int | None = None
        self._started = False
        self._paused = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._begin_gpu_segment()
        self._cpu_sampler.start()

    def pause(self) -> None:
        if not self._started or self._paused:
            return
        self._cpu_sampler.pause()
        self._capture_gpu_segment()
        self._paused = True

    def resume(self) -> None:
        if not self._started or not self._paused:
            return
        self._begin_gpu_segment()
        self._cpu_sampler.resume()
        self._paused = False

    def stop(self) -> tuple[int, int | None]:
        if not self._started:
            return 0, None
        if not self._paused:
            self._capture_gpu_segment()
        peak_cpu_memory_bytes = self._cpu_sampler.stop()
        self._started = False
        return peak_cpu_memory_bytes, self._peak_gpu_memory_bytes

    def _begin_gpu_segment(self) -> None:
        if self._cuda_device is None:
            return
        self._torch.cuda.synchronize(self._cuda_device)
        self._torch.cuda.reset_peak_memory_stats(self._cuda_device)

    def _capture_gpu_segment(self) -> None:
        if self._cuda_device is None:
            return
        self._torch.cuda.synchronize(self._cuda_device)
        segment_peak = int(self._torch.cuda.max_memory_allocated(self._cuda_device))
        current_peak = self._peak_gpu_memory_bytes or 0
        self._peak_gpu_memory_bytes = max(current_peak, segment_peak)


@contextmanager
def collect_training_efficiency(
    *,
    enabled: bool = True,
    device: Any = "cpu",
) -> Iterator[TrainingEfficiencyRecorder]:
    recorder = TrainingEfficiencyRecorder(enabled=enabled, run_device=device)
    token = _CURRENT_RECORDER.set(recorder)
    try:
        recorder.start_run_memory_measurement()
        yield recorder
    finally:
        try:
            recorder.stop_active_measurements()
        finally:
            try:
                recorder.stop_run_memory_measurement()
            finally:
                _CURRENT_RECORDER.reset(token)


@contextmanager
def suspend_efficiency_memory() -> Iterator[None]:
    recorder = _CURRENT_RECORDER.get()
    if recorder is None or not recorder.enabled:
        yield
        return

    recorder.suspend_run_memory_measurement()
    try:
        yield
    finally:
        recorder.resume_run_memory_measurement()


def start_train_eval_measurement(device: Any) -> TrainingMeasurement:
    return TrainingMeasurement(_CURRENT_RECORDER.get(), device)


class _PeakRSSSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        psutil = require_module("psutil", "pip install psutil")
        self._psutil = psutil
        self._process = psutil.Process()
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes = 0

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="vlm4cluster-rss-sampler", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()
        self._sample()

    def resume(self) -> None:
        self._sample()
        self._pause_event.clear()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if not self._pause_event.is_set():
            self._sample()
        return self._peak_bytes

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            if not self._pause_event.is_set():
                self._sample()

    def _sample(self) -> None:
        try:
            rss_bytes = int(self._process.memory_info().rss)
        except (self._psutil.Error, OSError):
            return
        try:
            children = self._process.children(recursive=True)
        except (self._psutil.Error, OSError):
            children = []
        for child in children:
            try:
                rss_bytes += int(child.memory_info().rss)
            except (self._psutil.Error, OSError):
                continue
        self._peak_bytes = max(self._peak_bytes, rss_bytes)
