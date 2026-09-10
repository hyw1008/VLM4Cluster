from __future__ import annotations

from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.subspace.common import (
    encode_evaluation_features,
    finalize_subspace_output,
    run_subspace_clustering,
    sparse_subspace_clustering_orthogonal_matching_pursuit,
    timed_run,
)
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.progress import get_progress_logger


SSC_OMP_BACKBONE_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10"): {"n_nonzero": 3},
    ("laion400m", "vitb32", "cifar20"): {"n_nonzero": 3},
    ("laion400m", "vitb32", "cifar100"): {"n_nonzero": 4},
    ("laion400m", "vitb32", "stl10"): {"n_nonzero": 4},
    ("laion400m", "vitb32", "imagenetdogs"): {"n_nonzero": 4},
    ("laion400m", "vitb32", "dtd"): {"n_nonzero": 7},
    ("laion400m", "vitb32", "ucf101"): {"n_nonzero": 4},
    ("laion400m", "vitb32", "aircraft"): {"n_nonzero": 6},
    ("laion400m", "vitb32", "cars"): {"n_nonzero": 15},
    ("laion400m", "vitb32", "flowers"): {"n_nonzero": 7},
    ("laion400m", "vitb32", "pets"): {"n_nonzero": 3},
    ("laion400m", "vitb16", "cifar10"): {"n_nonzero": 3},
    ("laion400m", "vitb16", "cifar20"): {"n_nonzero": 3},
    ("laion400m", "vitb16", "cifar100"): {"n_nonzero": 4},
    ("laion400m", "vitb16", "stl10"): {"n_nonzero": 3},
    ("laion400m", "vitb16", "imagenetdogs"): {"n_nonzero": 4},
    ("laion400m", "vitb16", "ucf101"): {"n_nonzero": 3},
    ("laion400m", "vitb16", "flowers"): {"n_nonzero": 6},
    ("laion400m", "vitb16", "pets"): {"n_nonzero": 3},
    ("laion400m", "vitl14", "cifar10"): {"n_nonzero": 3},
    ("laion400m", "vitl14", "cifar20"): {"n_nonzero": 3},
    ("laion400m", "vitl14", "cifar100"): {"n_nonzero": 6},
    ("laion400m", "vitl14", "stl10"): {"n_nonzero": 3},
    ("laion400m", "vitl14", "imagenet10"): {"n_nonzero": 3},
    ("laion400m", "vitl14", "imagenetdogs"): {"n_nonzero": 4},
    ("laion400m", "vitl14", "dtd"): {"n_nonzero": 4},
    ("laion400m", "vitl14", "pets"): {"n_nonzero": 7},
}


def _normalize_ssc_omp_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _resolve_ssc_omp_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_ssc_omp_profile_token(openclip_pretraining),
        _normalize_ssc_omp_profile_token(openclip_backbone),
        _normalize_ssc_omp_profile_token(dataset_name),
    )
    return dict(SSC_OMP_BACKBONE_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_ssc_omp_hyperparameter_profile_source(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> str | None:
    profile_key = (
        _normalize_ssc_omp_profile_token(openclip_pretraining),
        _normalize_ssc_omp_profile_token(openclip_backbone),
        _normalize_ssc_omp_profile_token(dataset_name),
    )
    if profile_key not in SSC_OMP_BACKBONE_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def run_ssc_omp_pipeline(inputs: MethodInputs, params: dict[str, Any]):
    progress = get_progress_logger("ssc_omp", params)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    source_dataset_name = str(params.get("source_dataset_name", inputs.image_dataset.name))
    bundle, _, source_dataset_name, evaluation_dataset, evaluation_split, domain_shift, features, labels = (
        encode_evaluation_features(
            inputs,
            params,
            method_name="SSC-OMP",
            cache_subdir="ssc_omp_cache",
        )
    )

    hyperparameter_defaults = _resolve_ssc_omp_hyperparameter_defaults(
        source_dataset_name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    hyperparameter_profile_source = _resolve_ssc_omp_hyperparameter_profile_source(
        source_dataset_name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    n_clusters = int(params.get("n_clusters", evaluation_dataset.num_classes))
    n_nonzero = int(params.get("n_nonzero", hyperparameter_defaults.get("n_nonzero", 10)))
    thr = float(params.get("thr", 1.0e-6))
    affinity = str(params.get("affinity", "symmetrize"))
    affinity_n_neighbors = int(params.get("affinity_n_neighbors", 3))
    random_state = params.get("random_state", params.get("seed"))
    n_init = int(params.get("n_init", 10))

    if n_clusters <= 0:
        raise ValueError("SSC-OMP requires n_clusters > 0.")
    if n_nonzero <= 0:
        raise ValueError("SSC-OMP requires n_nonzero > 0.")
    if thr <= 0:
        raise ValueError("SSC-OMP requires thr > 0.")
    if affinity_n_neighbors <= 0:
        raise ValueError("SSC-OMP requires affinity_n_neighbors > 0.")

    runtime_device = getattr(getattr(inputs, "runtime_config", None), "device", "cpu")
    efficiency_measurement = start_train_eval_measurement(runtime_device)
    with progress.stage("Computing SSC-OMP self-representation matrix"):
        representation_matrix, representation_time = timed_run(
            sparse_subspace_clustering_orthogonal_matching_pursuit,
            features.astype(np.float64, copy=False),
            n_nonzero=n_nonzero,
            thr=thr,
        )
    with progress.stage(f"Running spectral clustering with {n_clusters} cluster(s)"):
        predictions, clustering_time = timed_run(
            run_subspace_clustering,
            representation_matrix,
            n_clusters=n_clusters,
            affinity=affinity,
            affinity_n_neighbors=affinity_n_neighbors,
            random_state=None if random_state is None else int(random_state),
            n_init=n_init,
        )
    efficiency_measurement.stop()

    return finalize_subspace_output(
        method_name="ssc_omp",
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
            "n_nonzero": n_nonzero,
            "hyperparameter_profile": hyperparameter_profile_source,
            "thr": thr,
            "affinity": affinity,
            "affinity_n_neighbors": affinity_n_neighbors,
            "n_init": n_init,
            "representation_time_sec": representation_time,
            "clustering_time_sec": clustering_time,
            "evaluation_image_count": int(features.shape[0]),
            "representation_nnz": int(representation_matrix.nnz),
        },
    )
