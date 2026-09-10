from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm4cluster.utils.deps import require_module


def bool_param(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def save_checkpoint_enabled(params: dict[str, Any], dataset_name: str | None = None) -> bool:
    if dataset_name is not None and not is_full_imagenet_dataset(dataset_name):
        return False
    return bool_param(params.get("save_checkpoint", params.get("save_checkpoints")), False)


def is_full_imagenet_dataset(dataset_name: str | None) -> bool:
    if dataset_name is None:
        return False
    normalized = str(dataset_name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return normalized in {"imagenet", "imagenet1k", "ilsvrc2012"}


def save_checkpoint_enabled_for_dataset(params: dict[str, Any], dataset_name: str | None) -> bool:
    return save_checkpoint_enabled(params, dataset_name)


def load_checkpoint_enabled(
    params: dict[str, Any],
    dataset_name: str | None = None,
    *,
    default: bool = True,
) -> bool:
    if dataset_name is not None and not is_full_imagenet_dataset(dataset_name):
        return False
    if bool_param(params.get("force_retrain"), False):
        return False
    return bool_param(
        params.get(
            "load_checkpoint",
            params.get("reuse_checkpoint", params.get("reuse_checkpoints")),
        ),
        default,
    )


def load_checkpoint_enabled_for_dataset(
    params: dict[str, Any],
    dataset_name: str | None,
    default: bool = True,
) -> bool:
    if not is_full_imagenet_dataset(dataset_name):
        return False
    return load_checkpoint_enabled(params, dataset_name, default=default)


def checkpoint_interval(
    params: dict[str, Any],
    *,
    enabled: bool,
    default: int,
    aliases: tuple[str, ...] = ("save_every", "checkpoint_every"),
) -> int:
    if not enabled:
        return 0
    for key in aliases:
        if key in params and params[key] is not None:
            return int(params[key])
    return int(default)


def checkpoint_dir(base_dir: Path) -> Path:
    path = base_dir / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_file(base_dir: Path, filename: str, *, create: bool = False) -> Path:
    directory = checkpoint_dir(base_dir) if create else base_dir / "checkpoints"
    return directory / filename


def load_torch_checkpoint(path: Path, *, device=None) -> dict[str, Any]:
    torch = require_module("torch", "pip install torch")
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["map_location"] = device
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def first_existing_checkpoint(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def latest_checkpoint(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    paths = [path for path in directory.glob(pattern) if path.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)
