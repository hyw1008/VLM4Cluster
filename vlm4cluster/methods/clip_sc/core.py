from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.feature_cache import cache_file_path, cache_key, image_dataset_cache_key
from vlm4cluster.methods.subspace.common import (
    encode_evaluation_features,
    finalize_subspace_output,
    timed_run,
)
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import search_topk
from vlm4cluster.utils.progress import get_progress_logger


DEFAULT_CLIP_SC_ASSIGN_LABELS = "discretize"
DEFAULT_CLIP_SC_AFFINITY_KERNEL = "dot_product"
DEFAULT_CLIP_SC_GRAPH_K: int | None = None
DEFAULT_CLIP_SC_AFFINITY_TAU = 1.0
DEFAULT_CLIP_SC_RANDOM_STATE = 42
DEFAULT_CLIP_SC_N_INIT = 10


def run_clip_sc_pipeline(inputs: MethodInputs, params: dict[str, Any]):
    progress = get_progress_logger("clip_sc", params)
    _validate_formula_configuration(params)
    affinity_kernel = _resolve_affinity_kernel(params)
    bundle, output_dir, source_dataset_name, evaluation_dataset, evaluation_split, domain_shift, features, labels = (
        encode_evaluation_features(
            inputs,
            params,
            method_name="CLIP SC",
            cache_subdir="clip_sc_cache",
        )
    )

    n_clusters = int(params.get("n_clusters", evaluation_dataset.num_classes))
    assign_labels = str(params.get("assign_labels", DEFAULT_CLIP_SC_ASSIGN_LABELS))
    graph_k = _resolve_graph_k(params, sample_count=features.shape[0])
    affinity_tau = _resolve_affinity_tau(params) if affinity_kernel == "rbf" else None
    n_init = int(params.get("n_init", DEFAULT_CLIP_SC_N_INIT))
    random_state = _resolve_sklearn_random_state(params)

    if n_clusters <= 0:
        raise ValueError("CLIP-SC requires n_clusters > 0.")
    if n_clusters >= features.shape[0]:
        raise ValueError(
            f"CLIP-SC requires n_clusters < number of evaluation samples, got {n_clusters} and {features.shape[0]}."
        )
    if graph_k <= 0:
        raise ValueError("CLIP-SC requires graph_k > 0.")
    if affinity_tau is not None and affinity_tau <= 0:
        raise ValueError("CLIP-SC requires affinity_tau > 0.")

    effective_graph_k = min(graph_k, features.shape[0] - 1)
    full_graph = effective_graph_k == features.shape[0] - 1
    if affinity_kernel == "dot_product" and not full_graph:
        raise ValueError(
            "CLIP-SC dot_product uses the full sample Gram matrix (q=M-1). "
            "Remove graph_k/n_neighbors/q or set affinity_kernel=rbf for the mutual-neighbor graph."
        )
    affinity_cache_path = _resolve_affinity_cache_path(
        output_dir=output_dir,
        dataset_key=image_dataset_cache_key(evaluation_dataset),
        sample_count=features.shape[0],
        effective_graph_k=effective_graph_k,
        affinity_tau=affinity_tau,
        affinity_kernel=affinity_kernel,
    )
    graph_label = (
        "full dot-product affinity graph (zero diagonal)"
        if affinity_kernel == "dot_product"
        else "full RBF affinity graph"
        if full_graph
        else f"weighted mutual top-{graph_k} RBF affinity graph"
    )
    runtime_device = getattr(getattr(inputs, "runtime_config", None), "device", "cpu")
    efficiency_measurement = start_train_eval_measurement(runtime_device)
    with progress.stage(f"Building {graph_label}"):
        affinity_matrix, graph_metadata, graph_construction_time, affinity_cache_hit = (
            _load_or_build_affinity(
                features,
                graph_k=graph_k,
                affinity_tau=affinity_tau,
                cache_path=affinity_cache_path,
                force_recompute=bool(params.get("force_recompute_affinity", False)),
                progress=progress,
                affinity_kernel=affinity_kernel,
                device=runtime_device,
            )
        )

    with progress.stage(f"Running precomputed-affinity spectral clustering with {n_clusters} cluster(s)"):
        predictions, clustering_time = timed_run(
            _run_precomputed_spectral_clustering,
            affinity_matrix,
            n_clusters=n_clusters,
            assign_labels=assign_labels,
            n_init=n_init,
            random_state=random_state,
        )
    efficiency_measurement.stop()

    return finalize_subspace_output(
        method_name="clip_sc",
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
            "input_features": "common normalized raw OpenCLIP image features",
            "spectral_backend": "sklearn.cluster.SpectralClustering",
            "affinity": "precomputed",
            "affinity_kernel": affinity_kernel,
            "affinity_construction": (
                "full_dot_product" if affinity_kernel == "dot_product"
                else "full_rbf" if full_graph else "weighted_mutual_knn_rbf"
            ),
            "graph_mode": "full" if full_graph else "mutual_knn",
            "graph_k": graph_k,
            "q": graph_k,
            "affinity_tau": affinity_tau,
            "tau": affinity_tau,
            "gamma_equivalent": 1.0 / affinity_tau if affinity_tau is not None else None,
            "assign_labels": assign_labels,
            "n_init": n_init,
            "random_state": random_state,
            "evaluation_image_count": int(features.shape[0]),
            "graph_construction_time_sec": graph_construction_time,
            "affinity_cache_hit": affinity_cache_hit,
            "affinity_cache_path": str(affinity_cache_path),
            "clustering_time_sec": clustering_time,
            "evaluation_only_baseline": True,
            "runner_random_state_param": params.get("random_state"),
            **graph_metadata,
        },
    )


def _load_or_build_affinity(
    features: np.ndarray,
    *,
    graph_k: int,
    affinity_tau: float | None,
    cache_path: Path,
    force_recompute: bool,
    progress=None,
    affinity_kernel: str = DEFAULT_CLIP_SC_AFFINITY_KERNEL,
    device: str = "cpu",
):
    from scipy import sparse

    if cache_path.exists() and not force_recompute:
        if progress is not None:
            progress.log(f"Loading cached CLIP-SC affinity graph: {cache_path}")
        if graph_k >= features.shape[0] - 1:
            with np.load(cache_path) as payload:
                affinity_matrix = payload["affinity_matrix"]
        else:
            affinity_matrix = sparse.load_npz(cache_path).tocsr()
        if affinity_matrix.shape != (features.shape[0], features.shape[0]):
            raise ValueError(
                "Cached CLIP-SC affinity graph shape does not match the current feature matrix: "
                f"{affinity_matrix.shape} != {(features.shape[0], features.shape[0])}."
            )
        return (
            affinity_matrix,
            _summarize_affinity_graph(
                affinity_matrix,
                effective_graph_k=min(graph_k, features.shape[0] - 1),
            ),
            0.0,
            True,
        )

    if affinity_kernel == "dot_product":
        (affinity_matrix, graph_metadata), graph_construction_time = timed_run(
            _build_dot_product_affinity, features, device=device, progress=progress,
        )
    else:
        (affinity_matrix, graph_metadata), graph_construction_time = timed_run(
            _build_mutual_knn_rbf_affinity,
            features,
            graph_k=graph_k,
            affinity_tau=affinity_tau,
            progress=progress,
        )
    if graph_k >= features.shape[0] - 1:
        np.savez_compressed(cache_path, affinity_matrix=np.asarray(affinity_matrix, dtype=np.float32))
    else:
        sparse.save_npz(cache_path, affinity_matrix)
    if progress is not None:
        progress.log(f"Saved CLIP-SC affinity graph: {cache_path}")
    return affinity_matrix, graph_metadata, graph_construction_time, False


def _build_dot_product_affinity(features: np.ndarray, *, device: str = "cpu", progress=None):
    matrix = _normalize_rows(np.ascontiguousarray(features.astype("float32", copy=False)))
    sample_count = int(matrix.shape[0])
    if sample_count <= 1:
        raise ValueError("CLIP-SC requires at least two samples to build an affinity graph.")

    affinity_matrix = None
    tensor = None
    if str(device).startswith("cuda"):
        try:
            torch = require_module("torch", "pip install torch")
            if torch.cuda.is_available():
                with torch.no_grad():
                    tensor = torch.as_tensor(matrix, device=device)
                    affinity_matrix = np.empty((sample_count, sample_count), dtype=np.float32)
                    # Keep the dense graph on CPU; only a row block occupies GPU memory.
                    for start in range(0, sample_count, 1024):
                        stop = min(start + 1024, sample_count)
                        affinity_matrix[start:stop] = (tensor[start:stop] @ tensor.T).cpu().numpy()
                        if progress is not None:
                            progress.step("Dot-product affinity", start // 1024 + 1, (sample_count + 1023) // 1024)
                if progress is not None:
                    progress.log("Computed full dot-product affinity in CUDA row blocks")
        except RuntimeError as exc:
            affinity_matrix = None
            if progress is not None:
                progress.log(f"CUDA dot-product affinity unavailable ({type(exc).__name__}); using NumPy")
        finally:
            tensor = None
    if affinity_matrix is None:
        affinity_matrix = matrix @ matrix.T

    np.fill_diagonal(affinity_matrix, 0.0)
    return affinity_matrix, _summarize_affinity_graph(
        affinity_matrix, effective_graph_k=sample_count - 1,
    )


def _build_mutual_knn_rbf_affinity(
    features: np.ndarray,
    *,
    graph_k: int,
    affinity_tau: float,
    progress=None,
):
    from scipy import sparse

    matrix = _normalize_rows(np.ascontiguousarray(features.astype("float32", copy=False)))
    sample_count = int(matrix.shape[0])
    if sample_count <= 1:
        raise ValueError("CLIP-SC requires at least two samples to build an affinity graph.")
    if graph_k <= 0:
        raise ValueError("graph_k must be positive.")
    if affinity_tau <= 0:
        raise ValueError("affinity_tau must be positive.")

    effective_graph_k = min(graph_k, sample_count - 1)
    if effective_graph_k == sample_count - 1:
        affinity_matrix = matrix @ matrix.T
        np.clip(affinity_matrix, -1.0, 1.0, out=affinity_matrix)
        affinity_matrix *= 2.0
        affinity_matrix -= 2.0
        affinity_matrix /= affinity_tau
        np.exp(affinity_matrix, out=affinity_matrix)
        np.fill_diagonal(affinity_matrix, 0.0)
        return affinity_matrix, _summarize_affinity_graph(
            affinity_matrix,
            effective_graph_k=effective_graph_k,
        )

    similarities, neighbor_indices = search_topk(
        matrix,
        topk=effective_graph_k + 1,
        faiss_metric="ip",
        sklearn_metric="cosine",
        progress=progress,
        label="CLIP-SC nearest-neighbor search",
    )
    selected_indices = np.empty((sample_count, effective_graph_k), dtype=np.int64)
    selected_similarities = np.empty((sample_count, effective_graph_k), dtype=np.float32)
    for row_index in range(sample_count):
        valid = (neighbor_indices[row_index] >= 0) & (neighbor_indices[row_index] != row_index)
        row_indices = neighbor_indices[row_index][valid][:effective_graph_k]
        row_similarities = similarities[row_index][valid][:effective_graph_k]
        if row_indices.shape[0] != effective_graph_k:
            raise RuntimeError(
                "CLIP-SC nearest-neighbor search did not return enough non-self neighbors for "
                f"sample {row_index}: expected {effective_graph_k}, got {row_indices.shape[0]}."
            )
        selected_indices[row_index] = row_indices
        selected_similarities[row_index] = row_similarities

    squared_distances = np.maximum(
        2.0 - 2.0 * np.clip(selected_similarities, -1.0, 1.0),
        0.0,
    )
    weights = np.exp(-squared_distances / affinity_tau).astype(np.float32, copy=False)
    rows = np.repeat(np.arange(sample_count, dtype=np.int64), effective_graph_k)
    directed_affinity = sparse.csr_matrix(
        (weights.reshape(-1), (rows, selected_indices.reshape(-1))),
        shape=(sample_count, sample_count),
        dtype=np.float32,
    )
    directed_affinity.setdiag(0.0)
    directed_affinity.eliminate_zeros()

    mutual_affinity = directed_affinity.multiply(directed_affinity.T.sign()).tocsr()
    mutual_affinity = ((mutual_affinity + mutual_affinity.T) * 0.5).tocsr()
    mutual_affinity.setdiag(0.0)
    mutual_affinity.eliminate_zeros()

    return mutual_affinity, _summarize_affinity_graph(mutual_affinity, effective_graph_k=effective_graph_k)


def _summarize_affinity_graph(affinity_matrix, *, effective_graph_k: int) -> dict[str, Any]:
    from scipy import sparse

    sample_count = int(affinity_matrix.shape[0])
    if sparse.issparse(affinity_matrix):
        affinity_nnz = int(affinity_matrix.nnz)
        degrees = np.asarray(affinity_matrix.getnnz(axis=1)).reshape(-1)
        component_count = int(
            sparse.csgraph.connected_components(affinity_matrix, directed=False, return_labels=False)
        )
    else:
        affinity_nnz = int(np.count_nonzero(affinity_matrix))
        degrees = np.count_nonzero(affinity_matrix, axis=1)
        # Minimum degree >= M/2 guarantees connectivity without a dense-to-CSR copy.
        if np.all(degrees >= sample_count / 2):
            component_count = 1
        else:
            component_count = int(
                sparse.csgraph.connected_components(
                    sparse.csr_matrix(affinity_matrix),
                    directed=False,
                    return_labels=False,
                )
            )
    return {
        "effective_graph_k": effective_graph_k,
        "affinity_nnz": affinity_nnz,
        "affinity_density": float(affinity_nnz / (sample_count * sample_count)),
        "isolated_node_count": int(np.count_nonzero(degrees == 0)),
        "connected_component_count": component_count,
    }


def _resolve_affinity_cache_path(
    *,
    output_dir: Path,
    dataset_key: str,
    sample_count: int,
    effective_graph_k: int,
    affinity_tau: float | None,
    affinity_kernel: str = DEFAULT_CLIP_SC_AFFINITY_KERNEL,
) -> Path:
    if affinity_kernel == "dot_product":
        stem = cache_key(dataset_key, f"n{sample_count}", "full_dot_product_l2_zero_diag_v1")
        return cache_file_path(output_dir, f"{stem}__affinity", ".npz")
    graph_token = (
        "full_rbf_v1"
        if effective_graph_k == sample_count - 1
        else f"weighted_mutual_qnn_rbf_v2__q{effective_graph_k}"
    )
    stem = cache_key(
        dataset_key,
        f"n{sample_count}",
        graph_token,
        f"tau{affinity_tau}",
    )
    return cache_file_path(output_dir, f"{stem}__affinity", ".npz")


def _run_precomputed_spectral_clustering(
    affinity_matrix,
    *,
    n_clusters: int,
    assign_labels: str,
    n_init: int,
    random_state: int | None,
) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    degrees = np.asarray(affinity_matrix.sum(axis=1)).reshape(-1)
    if not np.isfinite(degrees).all() or np.any(degrees < 0):
        raise ValueError(
            "CLIP-SC cannot normalize an affinity with negative or non-finite degree sums. "
            "Raw dot products are preserved without clipping or shifting; "
            "affinity_kernel=rbf selects the nonnegative RBF graph."
        )

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels=assign_labels,
        n_init=n_init,
        random_state=random_state,
    )
    return clustering.fit_predict(affinity_matrix).astype(np.int64, copy=False)


def _resolve_affinity_kernel(params: dict[str, Any]) -> str:
    default = "rbf" if str(params.get("affinity", "")).strip().lower() == "rbf" else DEFAULT_CLIP_SC_AFFINITY_KERNEL
    kernel = str(params.get("affinity_kernel", default)).strip().lower()
    if kernel not in {"dot_product", "rbf"}:
        raise ValueError("CLIP-SC affinity_kernel must be dot_product or rbf.")
    if str(params.get("affinity", "")).strip().lower() == "rbf" and kernel != "rbf":
        raise ValueError("CLIP-SC affinity=rbf conflicts with affinity_kernel=dot_product.")
    return kernel


def _resolve_graph_k(params: dict[str, Any], *, sample_count: int) -> int:
    graph_mode = str(params.get("graph_mode", "")).strip().lower().replace("-", "_")
    if graph_mode == "full":
        return sample_count - 1
    value = params.get(
        "graph_k",
        params.get("n_neighbors", params.get("q", DEFAULT_CLIP_SC_GRAPH_K)),
    )
    return sample_count - 1 if value is None else int(value)


def _resolve_affinity_tau(params: dict[str, Any]) -> float:
    if "affinity_tau" in params:
        return float(params["affinity_tau"])
    if "tau" in params:
        return float(params["tau"])
    if "gamma" in params:
        gamma = float(params["gamma"])
        if gamma <= 0:
            raise ValueError("CLIP-SC requires gamma > 0 when using it as the inverse-temperature alias.")
        return 1.0 / gamma
    return DEFAULT_CLIP_SC_AFFINITY_TAU


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def _validate_formula_configuration(params: dict[str, Any]) -> None:
    _resolve_affinity_kernel(params)
    if "graph_mode" in params:
        graph_mode = str(params["graph_mode"]).strip().lower().replace("-", "_")
        if graph_mode not in {"auto", "full", "knn", "mutual", "mutual_knn"}:
            raise ValueError(
                "CLIP-SC graph_mode must be one of: auto, full, knn, mutual, mutual_knn."
            )
    if "affinity" in params:
        affinity = str(params["affinity"]).strip().lower()
        if affinity not in {"rbf", "precomputed"}:
            raise ValueError(
                "CLIP-SC uses a precomputed affinity; affinity must be precomputed or the legacy rbf alias."
            )


def _resolve_sklearn_random_state(params: dict[str, Any]) -> int | None:
    value = params.get(
        "spectral_random_state",
        params.get(
            "sc_random_state",
            params.get("random_state", params.get("seed", DEFAULT_CLIP_SC_RANDOM_STATE)),
        ),
    )
    return None if value is None else int(value)
