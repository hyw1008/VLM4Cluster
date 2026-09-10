from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Any, Iterator


_FALSE_VALUES = {"0", "false", "no", "off", "quiet"}
_CURRENT_ENABLED: bool | None = None


def _coerce_progress_enabled(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSE_VALUES


def is_progress_enabled(params: dict[str, Any] | None = None, *, default: bool = True) -> bool:
    env_value = os.environ.get("VLM4CLUSTER_PROGRESS")
    if env_value is not None and not _coerce_progress_enabled(env_value, default=default):
        return False
    enabled = _coerce_progress_enabled(env_value, default=default)
    if params is None:
        return enabled
    return _coerce_progress_enabled(params.get("progress"), default=enabled)


def get_progress_logger(
    name: str,
    params: dict[str, Any] | None = None,
    *,
    default: bool = True,
) -> "ProgressLogger":
    global _CURRENT_ENABLED
    inherited_default = _CURRENT_ENABLED if params is None and _CURRENT_ENABLED is not None else default
    enabled = is_progress_enabled(params, default=inherited_default)
    if params is not None:
        _CURRENT_ENABLED = enabled
    return ProgressLogger(name, enabled=enabled)


class ProgressLogger:
    def __init__(self, name: str, *, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"[{self.name}] {message}", flush=True)

    @contextmanager
    def stage(self, message: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start = time.monotonic()
        self.log(f"{message} ...")
        try:
            yield
        except Exception:
            self.log(f"{message} failed after {_format_seconds(time.monotonic() - start)}")
            raise
        self.log(f"{message} done in {_format_seconds(time.monotonic() - start)}")

    def epoch(
        self,
        label: str,
        current: int,
        total: int,
        *,
        every: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.step(label, current, total, noun="epoch", every=every, metrics=metrics)

    def step(
        self,
        label: str,
        current: int,
        total: int,
        *,
        noun: str = "step",
        every: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        interval = _resolve_interval(total, every)
        if current != 1 and current != total and current % interval != 0:
            return

        metric_text = ""
        if metrics:
            parts = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    parts.append(f"{key}={value:.4f}")
                else:
                    parts.append(f"{key}={value}")
            metric_text = " | " + ", ".join(parts)
        self.log(f"{label}: {noun} {current}/{total}{metric_text}")


def _resolve_interval(total: int, every: int | None) -> int:
    if every is not None and every > 0:
        return every
    if total <= 10:
        return 1
    return max(1, total // 10)


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"
