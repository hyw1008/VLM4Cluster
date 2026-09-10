from __future__ import annotations

from typing import Any


def resolve_source_default_cluster_num(
    params: dict[str, Any],
    train_dataset,
    *,
    method_name: str,
) -> tuple[int, str]:
    explicit = params.get("n_clusters")
    if explicit is not None:
        return int(explicit), "explicit"
    if train_dataset.num_classes <= 0:
        raise ValueError(f"{method_name} requires source train datasets with known class names to infer n_clusters.")
    return int(train_dataset.num_classes), "source_train_dataset"


def resolve_source_default_out_dim(
    params: dict[str, Any],
    train_dataset,
    *,
    method_name: str,
) -> tuple[int, str]:
    explicit_out_dim = params.get("out_dim")
    if explicit_out_dim is not None:
        return int(explicit_out_dim), "explicit_out_dim"
    explicit_clusters = params.get("n_clusters")
    if explicit_clusters is not None:
        return int(explicit_clusters), "explicit_n_clusters"
    if train_dataset.num_classes <= 0:
        raise ValueError(f"{method_name} requires source train datasets with known class names to infer out_dim.")
    return int(train_dataset.num_classes), "source_train_dataset"
