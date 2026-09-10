from __future__ import annotations

from typing import Any, Literal

import numpy as np

from vlm4cluster.utils.deps import require_module


DEFAULT_FAISS_QUERY_CHUNK_SIZE = 16384
FAISS_KMEANS_CACHE_TOKEN = "faiss_kmeans_v1"


def run_faiss_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    n_iter: int,
    n_redo: int,
    spherical: bool = False,
    random_state: int | None = None,
    prefer_gpu: bool = True,
    progress=None,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if n_redo <= 0:
        raise ValueError("n_redo must be positive.")

    matrix = np.ascontiguousarray(features.astype("float32", copy=False))
    if matrix.ndim != 2:
        raise ValueError("run_faiss_kmeans expects a 2D feature matrix.")
    if n_clusters > matrix.shape[0]:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed sample count ({matrix.shape[0]}).")

    try:
        faiss = require_module("faiss", "pip install faiss-gpu-cu12 --no-deps")
    except RuntimeError as exc:
        return _run_sklearn_kmeans(
            matrix,
            n_clusters=n_clusters,
            n_iter=n_iter,
            n_redo=n_redo,
            spherical=spherical,
            random_state=random_state,
            reason=f"FAISS unavailable ({exc.__class__.__name__})",
            progress=progress,
            label=label,
        )

    gpu_enabled, gpu_backend = _should_use_faiss_gpu(faiss, prefer_gpu=prefer_gpu)
    if gpu_enabled:
        try:
            return _train_faiss_kmeans(
                faiss,
                matrix,
                n_clusters=n_clusters,
                n_iter=n_iter,
                n_redo=n_redo,
                spherical=spherical,
                random_state=random_state,
                gpu=True,
                backend=gpu_backend,
                progress=progress,
                label=label,
            )
        except (AttributeError, RuntimeError) as exc:
            if progress is not None and label is not None:
                progress.log(f"{label}: GPU FAISS KMeans unavailable ({exc.__class__.__name__}), retrying with CPU FAISS")

    try:
        return _train_faiss_kmeans(
            faiss,
            matrix,
            n_clusters=n_clusters,
            n_iter=n_iter,
            n_redo=n_redo,
            spherical=spherical,
            random_state=random_state,
            gpu=False,
            backend="cpu faiss KMeans",
            progress=progress,
            label=label,
        )
    except (AttributeError, RuntimeError, ValueError) as exc:
        return _run_sklearn_kmeans(
            matrix,
            n_clusters=n_clusters,
            n_iter=n_iter,
            n_redo=n_redo,
            spherical=spherical,
            random_state=random_state,
            reason=f"CPU FAISS KMeans unavailable ({exc.__class__.__name__})",
            progress=progress,
            label=label,
        )


def search_topk(
    index_features: np.ndarray,
    *,
    topk: int,
    faiss_metric: Literal["ip", "l2"],
    query_features: np.ndarray | None = None,
    sklearn_metric: Literal["cosine", "euclidean"] | None = None,
    prefer_gpu: bool = True,
    query_chunk_size: int | None = None,
    progress=None,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if topk <= 0:
        raise ValueError("topk must be positive.")

    index_matrix = np.ascontiguousarray(index_features.astype("float32", copy=False))
    query_matrix = index_matrix if query_features is None else np.ascontiguousarray(query_features.astype("float32", copy=False))
    if index_matrix.ndim != 2 or query_matrix.ndim != 2:
        raise ValueError("search_topk expects 2D feature matrices.")
    if index_matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError(
            "index_features and query_features must share the same feature dimension: "
            f"{index_matrix.shape[1]} != {query_matrix.shape[1]}"
        )

    effective_sklearn_metric = sklearn_metric or _default_sklearn_metric(faiss_metric)
    effective_chunk_size = _resolve_query_chunk_size(query_matrix.shape[0], query_chunk_size)

    try:
        search_index, backend, _resources = _build_faiss_index(index_matrix, metric=faiss_metric, prefer_gpu=prefer_gpu)
        if progress is not None and label is not None:
            progress.log(f"{label}: using {backend}")
        scores, indices = _search_faiss_index(
            search_index,
            query_matrix,
            topk=topk,
            chunk_size=effective_chunk_size,
            progress=progress,
            label=label,
        )
        return scores.astype("float32", copy=False), indices.astype(np.int64, copy=False)
    except (AttributeError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        if progress is not None and label is not None:
            progress.log(f"{label}: FAISS path unavailable ({exc.__class__.__name__}), falling back to sklearn")

    sklearn_neighbors = require_module("sklearn.neighbors", "pip install scikit-learn")
    estimator = sklearn_neighbors.NearestNeighbors(n_neighbors=topk, metric=effective_sklearn_metric)
    estimator.fit(index_matrix)
    if progress is not None and label is not None:
        progress.log(f"{label}: using sklearn ({effective_sklearn_metric})")
    scores, indices = _search_sklearn_index(
        estimator,
        query_matrix,
        topk=topk,
        chunk_size=effective_chunk_size,
        faiss_metric=faiss_metric,
        progress=progress,
        label=label,
    )
    return scores.astype("float32", copy=False), indices.astype(np.int64, copy=False)


def _build_faiss_index(index_matrix: np.ndarray, *, metric: Literal["ip", "l2"], prefer_gpu: bool):
    faiss = require_module("faiss", "pip install faiss-gpu-cu12 --no-deps")
    if metric == "ip":
        cpu_index = faiss.IndexFlatIP(index_matrix.shape[1])
    elif metric == "l2":
        cpu_index = faiss.IndexFlatL2(index_matrix.shape[1])
    else:
        raise ValueError(f"Unsupported FAISS metric '{metric}'.")
    cpu_index.add(index_matrix)

    if not prefer_gpu:
        return cpu_index, "cpu faiss", None

    torch = require_module("torch", "pip install torch")
    if not torch.cuda.is_available():
        return cpu_index, "cpu faiss", None

    resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(resources, int(torch.cuda.current_device()), cpu_index)
    return gpu_index, f"gpu faiss (cuda:{torch.cuda.current_device()})", resources


def _search_faiss_index(index, query_matrix: np.ndarray, *, topk: int, chunk_size: int, progress=None, label: str | None = None):
    if query_matrix.shape[0] <= chunk_size:
        return index.search(query_matrix, topk)

    total_chunks = (query_matrix.shape[0] + chunk_size - 1) // chunk_size
    scores_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    for chunk_index, start in enumerate(range(0, query_matrix.shape[0], chunk_size), start=1):
        end = min(start + chunk_size, query_matrix.shape[0])
        chunk_scores, chunk_indices = index.search(query_matrix[start:end], topk)
        scores_parts.append(chunk_scores)
        indices_parts.append(chunk_indices)
        if progress is not None and label is not None:
            progress.step(label, chunk_index, total_chunks, noun="chunk")
    return np.concatenate(scores_parts, axis=0), np.concatenate(indices_parts, axis=0)


def _search_sklearn_index(
    estimator,
    query_matrix: np.ndarray,
    *,
    topk: int,
    chunk_size: int,
    faiss_metric: Literal["ip", "l2"],
    progress=None,
    label: str | None = None,
):
    if query_matrix.shape[0] <= chunk_size:
        distances, indices = estimator.kneighbors(query_matrix, n_neighbors=topk)
        return _convert_sklearn_distances(distances, faiss_metric), indices

    total_chunks = (query_matrix.shape[0] + chunk_size - 1) // chunk_size
    score_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for chunk_index, start in enumerate(range(0, query_matrix.shape[0], chunk_size), start=1):
        end = min(start + chunk_size, query_matrix.shape[0])
        distances, indices = estimator.kneighbors(query_matrix[start:end], n_neighbors=topk)
        score_parts.append(_convert_sklearn_distances(distances, faiss_metric))
        index_parts.append(indices)
        if progress is not None and label is not None:
            progress.step(label, chunk_index, total_chunks, noun="chunk")
    return np.concatenate(score_parts, axis=0), np.concatenate(index_parts, axis=0)


def _convert_sklearn_distances(distances: np.ndarray, faiss_metric: Literal["ip", "l2"]) -> np.ndarray:
    if faiss_metric == "ip":
        return 1.0 - distances.astype("float32", copy=False)
    return distances.astype("float32", copy=False)


def _default_sklearn_metric(faiss_metric: Literal["ip", "l2"]) -> Literal["cosine", "euclidean"]:
    if faiss_metric == "ip":
        return "cosine"
    if faiss_metric == "l2":
        return "euclidean"
    raise ValueError(f"Unsupported FAISS metric '{faiss_metric}'.")


def _resolve_query_chunk_size(query_count: int, explicit_chunk_size: int | None) -> int:
    if explicit_chunk_size is not None:
        if explicit_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when provided.")
        return explicit_chunk_size
    if query_count <= DEFAULT_FAISS_QUERY_CHUNK_SIZE:
        return query_count
    return DEFAULT_FAISS_QUERY_CHUNK_SIZE


def _should_use_faiss_gpu(faiss, *, prefer_gpu: bool) -> tuple[bool, str]:
    if not prefer_gpu:
        return False, "cpu faiss KMeans"
    if not hasattr(faiss, "StandardGpuResources"):
        return False, "cpu faiss KMeans"

    torch = require_module("torch", "pip install torch")
    if not torch.cuda.is_available():
        return False, "cpu faiss KMeans"
    return True, f"gpu faiss KMeans (cuda:{torch.cuda.current_device()})"


def _train_faiss_kmeans(
    faiss,
    matrix: np.ndarray,
    *,
    n_clusters: int,
    n_iter: int,
    n_redo: int,
    spherical: bool,
    random_state: int | None,
    gpu: bool,
    backend: str,
    progress=None,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs: dict[str, Any] = {
        "niter": n_iter,
        "nredo": n_redo,
        "spherical": spherical,
        "gpu": gpu,
    }
    if random_state is not None:
        kwargs["seed"] = int(random_state)
    if progress is not None and label is not None:
        progress.log(f"{label}: using {backend}")

    kmeans = faiss.Kmeans(matrix.shape[1], n_clusters, **kwargs)
    kmeans.train(matrix)
    _, indices = kmeans.index.search(matrix, 1)
    return indices.reshape(-1).astype(np.int64, copy=False), kmeans.centroids.astype("float32", copy=False)


def _run_sklearn_kmeans(
    matrix: np.ndarray,
    *,
    n_clusters: int,
    n_iter: int,
    n_redo: int,
    spherical: bool,
    random_state: int | None,
    reason: str,
    progress=None,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if progress is not None and label is not None:
        progress.log(f"{label}: {reason}, falling back to sklearn KMeans")
    sklearn_cluster = require_module("sklearn.cluster", "pip install scikit-learn")
    estimator = sklearn_cluster.KMeans(
        n_clusters=n_clusters,
        n_init=n_redo,
        max_iter=n_iter,
        random_state=random_state,
    )
    assignments = estimator.fit_predict(matrix).astype(np.int64, copy=False)
    centers = estimator.cluster_centers_.astype("float32", copy=False)
    if spherical:
        centers = _normalize_rows(centers)
    return assignments, centers


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms
