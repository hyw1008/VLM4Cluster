from __future__ import annotations

from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.subspace.common import encode_evaluation_features, finalize_subspace_output, timed_run
from vlm4cluster.methods.tac.core import _run_kmeans
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.progress import get_progress_logger


def run_clip_kmeans_pipeline(inputs: MethodInputs, params: dict[str, Any]):
    progress = get_progress_logger("clip_kmeans", params)
    bundle, _, source_dataset_name, evaluation_dataset, evaluation_split, domain_shift, features, labels = (
        encode_evaluation_features(
            inputs,
            params,
            method_name="CLIP KMeans",
            cache_subdir="clip_kmeans_cache",
        )
    )

    n_clusters = int(params.get("n_clusters", evaluation_dataset.num_classes))
    kmeans_niter = int(params.get("kmeans_niter", 300))
    kmeans_nredo = int(params.get("kmeans_nredo", 20))
    random_state = params.get("random_state", params.get("seed"))

    if n_clusters <= 0:
        raise ValueError("CLIP-KMeans requires n_clusters > 0.")

    runtime_device = getattr(getattr(inputs, "runtime_config", None), "device", "cpu")
    efficiency_measurement = start_train_eval_measurement(runtime_device)
    with progress.stage(f"Running CLIP K-Means with {n_clusters} cluster(s)"):
        predictions, clustering_time = timed_run(
            _run_kmeans,
            features,
            cluster_num=n_clusters,
            n_iter=kmeans_niter,
            n_redo=kmeans_nredo,
            random_state=None if random_state is None else int(random_state),
        )
    efficiency_measurement.stop()

    return finalize_subspace_output(
        method_name="clip_kmeans",
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
            "kmeans_niter": kmeans_niter,
            "kmeans_nredo": kmeans_nredo,
            "random_state": random_state,
            "evaluation_image_count": int(features.shape[0]),
            "clustering_time_sec": clustering_time,
            "evaluation_only_baseline": True,
        },
    )
