from __future__ import annotations

from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.subspace.common import (
    DEFAULT_ACTIVE_SUPPORT_PARAMS,
    elastic_net_subspace_clustering,
    encode_evaluation_features,
    finalize_subspace_output,
    run_subspace_clustering,
    timed_run,
)
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.progress import get_progress_logger


ENSC_DATASET_HYPERPARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "imagenet10": {
        "gamma": 25.0,
        "n_nonzero": 20,
        "n_init": 100,
        "active_support_params": {
            "support_size": 100,
        },
    },
    "imagenetdogs": {
        "gamma": 50.0,
        "n_nonzero": 10,
        "n_init": 100,
        "active_support_params": {
            "support_size": 75,
        },
    },
    "ucf101": {
        "gamma": 25.0,
        "n_nonzero": 20,
        "n_init": 100,
        "active_support_params": {
            "support_size": 100,
        },
    },
}

ENSC_BACKBONE_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10"): {"gamma": 100.0},
    ("laion400m", "vitb32", "cifar100"): {"gamma": 100.0},
    ("laion400m", "vitb32", "stl10"): {"gamma": 100.0},
    ("laion400m", "vitb32", "imagenetdogs"): {"gamma": 25.0},
    ("laion400m", "vitb32", "dtd"): {"gamma": 200.0},
    ("laion400m", "vitb32", "places365standard"): {"gamma": 100.0},
    ("laion400m", "vitb32", "flowers"): {"n_nonzero": 10},
    ("laion400m", "vitb16", "cifar100"): {"gamma": 100.0},
    ("laion400m", "vitb16", "stl10"): {"gamma": 100.0},
    ("laion400m", "vitb16", "imagenet10"): {"gamma": 100.0},
    ("laion400m", "vitb16", "imagenetdogs"): {"gamma": 25.0},
    ("laion400m", "vitb16", "flowers"): {"n_nonzero": 10},
    ("laion400m", "vitl14", "cifar100"): {"gamma": 100.0},
    ("laion400m", "vitl14", "imagenet10"): {"gamma": 50.0},
    ("laion400m", "vitl14", "imagenetdogs"): {"gamma": 25.0},
    ("laion400m", "vitl14", "dtd"): {"gamma": 200.0},
    ("laion400m", "vitl14", "flowers"): {"n_nonzero": 10},
}


def _normalize_ensc_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _normalize_ensc_profile_token(name: str) -> str:
    return _normalize_ensc_dataset_name(name).replace("/", "")


def _uses_ensc_generic_openclip_defaults(openclip_pretraining: str | None) -> bool:
    return (
        openclip_pretraining is not None
        and _normalize_ensc_profile_token(openclip_pretraining) == "siglip"
    )


def _resolve_ensc_backbone_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> dict[str, Any]:
    if _uses_ensc_generic_openclip_defaults(openclip_pretraining):
        return {}
    profile_key = (
        _normalize_ensc_profile_token(openclip_pretraining),
        _normalize_ensc_profile_token(openclip_backbone),
        _normalize_ensc_dataset_name(dataset_name),
    )
    return dict(ENSC_BACKBONE_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_ensc_backbone_hyperparameter_profile_source(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> str | None:
    if _uses_ensc_generic_openclip_defaults(openclip_pretraining):
        return None
    profile_key = (
        _normalize_ensc_profile_token(openclip_pretraining),
        _normalize_ensc_profile_token(openclip_backbone),
        _normalize_ensc_dataset_name(dataset_name),
    )
    if profile_key not in ENSC_BACKBONE_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_ensc_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
    openclip_backbone: str | None = None,
) -> dict[str, Any]:
    if _uses_ensc_generic_openclip_defaults(openclip_pretraining):
        return {}
    defaults = ENSC_DATASET_HYPERPARAMETER_DEFAULTS.get(_normalize_ensc_dataset_name(dataset_name), {})
    resolved = dict(defaults)
    if "active_support_params" in resolved:
        resolved["active_support_params"] = dict(resolved["active_support_params"])
    if openclip_pretraining is not None and openclip_backbone is not None:
        resolved.update(
            _resolve_ensc_backbone_hyperparameter_defaults(
                dataset_name,
                openclip_pretraining=openclip_pretraining,
                openclip_backbone=openclip_backbone,
            )
        )
    return resolved


def _resolve_ensc_param(
    params: dict[str, Any],
    dataset_defaults: dict[str, Any],
    name: str,
    fallback: Any,
) -> Any:
    value = params.get(name)
    if value is not None:
        return value
    return dataset_defaults.get(name, fallback)


def _resolve_active_support_params(
    active_support: bool,
    active_support_params: Any,
    dataset_active_support_params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    defaults = dict(DEFAULT_ACTIVE_SUPPORT_PARAMS)
    if dataset_active_support_params is not None:
        defaults.update(dataset_active_support_params)
    if active_support_params is None:
        return defaults if active_support else None
    if not isinstance(active_support_params, dict):
        raise TypeError("EnSC expects active_support_params to be a mapping when provided.")
    if not active_support:
        return dict(active_support_params)
    return {**defaults, **active_support_params}


def run_ensc_pipeline(inputs: MethodInputs, params: dict[str, Any]):
    progress = get_progress_logger("ensc", params)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    source_dataset_name = str(params.get("source_dataset_name", inputs.image_dataset.name))
    bundle, _, source_dataset_name, evaluation_dataset, evaluation_split, domain_shift, features, labels = (
        encode_evaluation_features(
            inputs,
            params,
            method_name="EnSC",
            cache_subdir="ensc_cache",
        )
    )

    dataset_defaults = _resolve_ensc_hyperparameter_defaults(
        source_dataset_name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    backbone_profile_source = _resolve_ensc_backbone_hyperparameter_profile_source(
        source_dataset_name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    n_clusters = int(params.get("n_clusters", evaluation_dataset.num_classes))
    affinity = str(params.get("affinity", "symmetrize"))
    random_state = params.get("random_state", params.get("seed"))
    n_init = int(_resolve_ensc_param(params, dataset_defaults, "n_init", 20))
    gamma = float(_resolve_ensc_param(params, dataset_defaults, "gamma", 50.0))
    gamma_nz = bool(params.get("gamma_nz", True))
    tau = float(params.get("tau", 1.0))
    algorithm = str(params.get("algorithm", "lasso_lars"))
    active_support = bool(params.get("active_support", True))
    active_support_params = params.get("active_support_params")
    n_nonzero = int(_resolve_ensc_param(params, dataset_defaults, "n_nonzero", 50))
    active_support_params = _resolve_active_support_params(
        active_support,
        active_support_params,
        dataset_defaults.get("active_support_params"),
    )

    if n_clusters <= 0:
        raise ValueError("EnSC requires n_clusters > 0.")
    if gamma <= 0:
        raise ValueError("EnSC requires gamma > 0.")
    if n_nonzero <= 0:
        raise ValueError("EnSC requires n_nonzero > 0.")

    runtime_device = getattr(getattr(inputs, "runtime_config", None), "device", "cpu")
    efficiency_measurement = start_train_eval_measurement(runtime_device)
    with progress.stage("Computing EnSC elastic-net self-representation matrix"):
        representation_matrix, representation_time = timed_run(
            elastic_net_subspace_clustering,
            features.astype(np.float64, copy=True),
            gamma=gamma,
            gamma_nz=gamma_nz,
            tau=tau,
            algorithm=algorithm,
            active_support=active_support,
            active_support_params=active_support_params,
            n_nonzero=n_nonzero,
        )
    with progress.stage(f"Running spectral clustering with {n_clusters} cluster(s)"):
        predictions, clustering_time = timed_run(
            run_subspace_clustering,
            representation_matrix,
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=None if random_state is None else int(random_state),
            n_init=n_init,
        )
    efficiency_measurement.stop()

    return finalize_subspace_output(
        method_name="ensc",
        predictions=np.asarray(predictions, dtype=np.int64),
        evaluation_labels=None if labels is None else np.asarray(labels, dtype=np.int64),
        evaluation_split=evaluation_split,
        metadata={
            "source_dataset": source_dataset_name,
            "evaluation_dataset": evaluation_dataset.name,
            "domain_shift": domain_shift,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "n_clusters": n_clusters,
            "affinity": affinity,
            "n_init": n_init,
            "gamma": gamma,
            "gamma_nz": gamma_nz,
            "tau": tau,
            "algorithm": algorithm,
            "active_support": active_support,
            "active_support_params": active_support_params,
            "dataset_default_params": dataset_defaults,
            "backbone_hyperparameter_profile": backbone_profile_source,
            "n_nonzero": n_nonzero,
            "representation_time_sec": representation_time,
            "clustering_time_sec": clustering_time,
            "evaluation_image_count": int(features.shape[0]),
            "representation_nnz": int(representation_matrix.nnz),
        },
    )
