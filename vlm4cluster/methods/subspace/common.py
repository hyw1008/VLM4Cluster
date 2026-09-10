from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.feature_cache import method_feature_cache_dir, resolve_benchmark_data_root
from vlm4cluster.methods.tac.core import (
    _build_named_split_dataset,
    _encode_image_dataset,
    _is_imagenet_variant_dataset,
    _normalize_dataset_name,
    default_evaluation_split,
)
from vlm4cluster.models import load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.faiss_utils import run_faiss_kmeans
from vlm4cluster.utils.progress import get_progress_logger


DEFAULT_ACTIVE_SUPPORT_PARAMS = {
    "support_init": "knn",
    "support_size": 100,
    "maxiter": 40,
}


@dataclass(slots=True)
class SubspaceClusteringOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


def resolve_image_only_evaluation_dataset(
    inputs: MethodInputs,
    params: dict[str, Any],
    *,
    method_name: str,
):
    source_dataset_name = str(params.get("source_dataset_name", inputs.image_dataset.name))
    target_dataset_name = params.get("target_dataset_name", params.get("target_dataset"))
    if target_dataset_name is None:
        evaluation_dataset_name = source_dataset_name
        domain_shift = False
    else:
        evaluation_dataset_name = str(target_dataset_name)
        domain_shift = True
        if _is_imagenet_variant_dataset(evaluation_dataset_name) and _normalize_dataset_name(source_dataset_name) != "imagenet":
            raise ValueError(
                f"{method_name} domain-shift protocol requires ImageNet as the source dataset when the target is "
                f"an ImageNet variant, got source='{source_dataset_name}' and target='{evaluation_dataset_name}'."
            )

    evaluation_split_default = default_evaluation_split(evaluation_dataset_name)
    evaluation_split = str(params.get("target_test_split", params.get("test_split", evaluation_split_default)))
    evaluation_dataset = _build_named_split_dataset(
        inputs,
        evaluation_dataset_name,
        evaluation_split,
        params.get("max_test_samples"),
    )
    return source_dataset_name, evaluation_dataset, evaluation_split, domain_shift


def encode_evaluation_features(
    inputs: MethodInputs,
    params: dict[str, Any],
    *,
    method_name: str,
    cache_subdir: str,
):
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger(method_name.lower().replace(" ", "_").replace("-", "_"), params)

    device = torch.device(inputs.runtime_config.device)
    progress.log(
        f"Loading OpenCLIP model '{params.get('openclip_backbone', 'ViT-B/32')}' "
        f"pretrained on '{params.get('openclip_pretraining', 'LAION400M')}'"
    )
    bundle = load_openclip_bundle(
        str(params.get("openclip_pretraining", "LAION400M")),
        str(params.get("openclip_backbone", "ViT-B/32")),
        str(device),
    )
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        cache_subdir.removesuffix("_cache"),
        bundle.spec.cache_key,
    )
    source_dataset_name, evaluation_dataset, evaluation_split, domain_shift = resolve_image_only_evaluation_dataset(
        inputs,
        params,
        method_name=method_name,
    )
    progress.log(
        f"Resolved evaluation protocol: source={source_dataset_name}, "
        f"eval={evaluation_dataset.name}/{evaluation_split}"
    )
    image_batch_size = int(params.get("image_batch_size", 256))
    with progress.stage(f"Encoding evaluation image split '{evaluation_dataset.name}/{evaluation_split}'"):
        features, labels = _encode_image_dataset(
            dataset=evaluation_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    return bundle, output_dir, source_dataset_name, evaluation_dataset, evaluation_split, domain_shift, features, labels


def representation_to_affinity(
    representation_matrix,
    affinity: str,
    *,
    affinity_n_neighbors: int = 3,
):
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import normalize

    normalized_representation_matrix = normalize(representation_matrix, "l2")
    if affinity == "symmetrize":
        return 0.5 * (np.absolute(normalized_representation_matrix) + np.absolute(normalized_representation_matrix.T))
    if affinity == "nearest_neighbors":
        if affinity_n_neighbors <= 0:
            raise ValueError("affinity_n_neighbors must be positive.")
        sample_count = normalized_representation_matrix.shape[0]
        if affinity_n_neighbors >= sample_count:
            raise ValueError(
                f"affinity_n_neighbors ({affinity_n_neighbors}) must be smaller than sample count ({sample_count})."
            )
        neighbors_graph = kneighbors_graph(
            normalized_representation_matrix,
            affinity_n_neighbors,
            mode="connectivity",
            include_self=False,
        )
        return 0.5 * (neighbors_graph + neighbors_graph.T)
    raise ValueError(f"Unknown affinity mode: {affinity}")


def spectral_clustering_from_affinity(
    affinity_matrix,
    *,
    n_clusters: int,
    random_state: int | None,
    n_init: int,
) -> np.ndarray:
    from scipy import sparse
    from sklearn.preprocessing import normalize
    from sklearn.utils import check_random_state, check_symmetric

    seed = int(random_state) if isinstance(random_state, (int, np.integer)) else None
    affinity_matrix = check_symmetric(affinity_matrix)
    check_random_state(random_state)
    laplacian = sparse.csgraph.laplacian(affinity_matrix, normed=True)
    _, eigenvectors = sparse.linalg.eigsh(
        sparse.identity(laplacian.shape[0]) - laplacian,
        k=n_clusters,
        sigma=None,
        which="LA",
    )
    embedding = normalize(eigenvectors)
    labels, _ = run_faiss_kmeans(
        embedding,
        n_clusters=n_clusters,
        n_iter=300,
        n_redo=n_init,
        spherical=False,
        random_state=seed,
    )
    return labels


def sparse_subspace_clustering_orthogonal_matching_pursuit(
    features: np.ndarray,
    *,
    n_nonzero: int,
    thr: float,
):
    from scipy import sparse

    n_samples = features.shape[0]
    rows = np.zeros(n_samples * n_nonzero, dtype=np.int64)
    cols = np.zeros(n_samples * n_nonzero, dtype=np.int64)
    vals = np.zeros(n_samples * n_nonzero, dtype=np.float64)
    curr_pos = 0

    for index in range(n_samples):
        residual = features[index, :].copy()
        support = np.empty(shape=(0,), dtype=np.int64)
        residual_norm_thr = np.linalg.norm(features[index, :]) * thr
        coefficients = np.empty(shape=(0,), dtype=np.float64)
        for _ in range(n_nonzero):
            coherence = np.absolute(np.matmul(residual, features.T))
            coherence[index] = 0.0
            support = np.append(support, np.argmax(coherence))
            coefficients = np.linalg.lstsq(features[support, :].T, features[index, :].T, rcond=None)[0]
            residual = features[index, :] - np.matmul(coefficients.T, features[support, :])
            if np.sum(residual**2) < residual_norm_thr:
                break

        rows[curr_pos : curr_pos + len(support)] = index
        cols[curr_pos : curr_pos + len(support)] = support
        vals[curr_pos : curr_pos + len(support)] = coefficients
        curr_pos += len(support)

    return sparse.csr_matrix((vals[:curr_pos], (rows[:curr_pos], cols[:curr_pos])), shape=(n_samples, n_samples))


def active_support_elastic_net(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    *,
    tau: float,
    algorithm: str,
    support_init: str = DEFAULT_ACTIVE_SUPPORT_PARAMS["support_init"],
    support_size: int = DEFAULT_ACTIVE_SUPPORT_PARAMS["support_size"],
    maxiter: int = DEFAULT_ACTIVE_SUPPORT_PARAMS["maxiter"],
) -> np.ndarray:
    from sklearn.decomposition import sparse_encode

    n_samples = features.shape[0]
    if n_samples <= support_size:
        support = np.arange(n_samples, dtype=np.int64)
    else:
        if support_init == "L2":
            l2_solution = np.linalg.solve(np.identity(target.shape[1]) * alpha + np.dot(features.T, features), target.T)
            c0 = np.dot(features, l2_solution)[:, 0]
            support = np.argpartition(-np.abs(c0), support_size)[:support_size]
        elif support_init == "knn":
            support = np.argpartition(-np.abs(np.dot(target, features.T)[0]), support_size)[:support_size]
        else:
            raise ValueError(f"Unknown support_init: {support_init}")

    curr_obj = float("inf")
    cs = np.zeros((1, len(support)), dtype=np.float64)
    for _ in range(maxiter):
        features_support = features[support, :]
        if algorithm == "spams":
            spams = require_module("spams", "pip install spams")
            cs = spams.lasso(
                np.asfortranarray(target.T),
                D=np.asfortranarray(features_support.T),
                lambda1=tau * alpha,
                lambda2=(1.0 - tau) * alpha,
            )
            cs = np.asarray(cs.todense()).T
        else:
            cs = sparse_encode(target, features_support, algorithm=algorithm, alpha=alpha)

        delta = (target - np.dot(cs, features_support)) / alpha
        objective = (
            tau * np.sum(np.abs(cs[0]))
            + (1.0 - tau) / 2.0 * np.sum(np.power(cs[0], 2.0))
            + alpha / 2.0 * np.sum(np.power(delta, 2.0))
        )
        if curr_obj - objective < 1.0e-10 * curr_obj:
            break
        curr_obj = objective

        coherence = np.abs(np.dot(delta, features.T))[0]
        coherence[support] = 0.0
        added_support = np.nonzero(coherence > tau + 1.0e-10)[0]
        if added_support.size == 0:
            break

        active_support = support[np.abs(cs[0]) > 1.0e-10]
        if active_support.size > 0.8 * support_size:
            support_size = min(round(max(active_support.size, support_size) * 1.1), n_samples)

        if added_support.size + active_support.size > support_size:
            order = np.argpartition(-coherence[added_support], support_size - active_support.size)[
                : support_size - active_support.size
            ]
            added_support = added_support[order]

        support = np.concatenate([active_support, added_support])

    coefficients = np.zeros(n_samples, dtype=np.float64)
    coefficients[support] = cs[0]
    return coefficients


def elastic_net_subspace_clustering(
    features: np.ndarray,
    *,
    gamma: float,
    gamma_nz: bool,
    tau: float,
    algorithm: str,
    active_support: bool,
    active_support_params: dict[str, Any] | None,
    n_nonzero: int,
):
    from scipy import sparse
    from sklearn.decomposition import sparse_encode

    if algorithm in ("lasso_lars", "lasso_cd") and tau < 1.0 - 1.0e-10:
        warnings.warn(f"algorithm {algorithm} cannot handle tau smaller than 1. Using tau = 1")
        tau = 1.0

    active_support_params = {} if active_support and active_support_params is None else active_support_params
    n_samples = features.shape[0]
    rows = np.zeros(n_samples * n_nonzero, dtype=np.int64)
    cols = np.zeros(n_samples * n_nonzero, dtype=np.int64)
    vals = np.zeros(n_samples * n_nonzero, dtype=np.float64)
    curr_pos = 0

    for index in range(n_samples):
        target = features[index, :].copy().reshape(1, -1)
        features[index, :] = 0.0

        if algorithm in ("lasso_lars", "lasso_cd", "spams"):
            if gamma_nz:
                alpha0 = np.amax(np.absolute(np.dot(features, target.T))) / tau
                alpha = alpha0 / gamma
            else:
                alpha = 1.0 / gamma

            if active_support:
                coefficients = active_support_elastic_net(
                    features,
                    target,
                    alpha,
                    tau=tau,
                    algorithm=algorithm,
                    **active_support_params,
                )
            else:
                if algorithm == "spams":
                    spams = require_module("spams", "pip install spams")
                    coefficients = spams.lasso(
                        np.asfortranarray(target.T),
                        D=np.asfortranarray(features.T),
                        lambda1=tau * alpha,
                        lambda2=(1.0 - tau) * alpha,
                    )
                    coefficients = np.asarray(coefficients.todense()).T[0]
                else:
                    coefficients = sparse_encode(target, features, algorithm=algorithm, alpha=alpha)[0]
        else:
            warnings.warn(f"algorithm {algorithm} not found")
            coefficients = np.zeros(n_samples, dtype=np.float64)

        nonzero_index = np.flatnonzero(coefficients)
        if nonzero_index.size > n_nonzero:
            nonzero_index = nonzero_index[np.argsort(-np.absolute(coefficients[nonzero_index]))[:n_nonzero]]
        rows[curr_pos : curr_pos + len(nonzero_index)] = index
        cols[curr_pos : curr_pos + len(nonzero_index)] = nonzero_index
        vals[curr_pos : curr_pos + len(nonzero_index)] = coefficients[nonzero_index]
        curr_pos += len(nonzero_index)
        features[index, :] = target[0]

    return sparse.csr_matrix((vals[:curr_pos], (rows[:curr_pos], cols[:curr_pos])), shape=(n_samples, n_samples))


def run_subspace_clustering(
    representation_matrix,
    *,
    n_clusters: int,
    affinity: str,
    affinity_n_neighbors: int = 3,
    random_state: int | None,
    n_init: int,
) -> np.ndarray:
    affinity_matrix = representation_to_affinity(
        representation_matrix,
        affinity,
        affinity_n_neighbors=affinity_n_neighbors,
    )
    return spectral_clustering_from_affinity(
        affinity_matrix,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )


def finalize_subspace_output(
    *,
    method_name: str,
    predictions: np.ndarray,
    evaluation_labels: np.ndarray | None,
    evaluation_split: str,
    metadata: dict[str, Any],
) -> SubspaceClusteringOutputs:
    return SubspaceClusteringOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=evaluation_labels.tolist() if evaluation_labels is not None else None,
        evaluation_split=evaluation_split,
        metadata=metadata,
    )


def timed_run(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start
