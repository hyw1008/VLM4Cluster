from __future__ import annotations

from typing import Type

from vlm4cluster.methods.base import ClusteringMethod


METHODS: dict[str, Type[ClusteringMethod]] = {}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def register_method(cls: Type[ClusteringMethod]) -> Type[ClusteringMethod]:
    METHODS[cls.name] = cls
    return cls


def get_method(name: str) -> Type[ClusteringMethod]:
    normalized = _normalize_name(name)
    for key, method in METHODS.items():
        if _normalize_name(key) == normalized:
            return method
    try:
        return METHODS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown clustering method '{name}'.") from exc


def list_methods() -> list[Type[ClusteringMethod]]:
    return [METHODS[key] for key in sorted(METHODS)]
