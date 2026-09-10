from __future__ import annotations


INTERNAL_CLUSTERING_METRICS = ("sil", "dbi", "chi")


def clustering_accuracy(labels: list[int], predictions: list[int]) -> float:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import contingency_matrix

    contingency = contingency_matrix(labels, predictions)
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / contingency.sum())


def evaluate_clustering(labels: list[int] | None, predictions: list[int], metrics: list[str]) -> dict[str, float]:
    if labels is None:
        return {}

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    results: dict[str, float] = {}
    for metric in metrics:
        if metric == "nmi":
            results["nmi"] = float(normalized_mutual_info_score(labels, predictions))
        elif metric == "ari":
            results["ari"] = float(adjusted_rand_score(labels, predictions))
        elif metric == "acc":
            results["acc"] = clustering_accuracy(labels, predictions)
        else:
            raise ValueError(f"Unknown evaluation metric '{metric}'.")
    return results


def normalize_internal_evaluation_features(features):
    import numpy as np

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Internal clustering metrics require a 2D feature matrix, got shape {matrix.shape}.")
    if matrix.shape[0] < 2:
        raise ValueError("Internal clustering metrics require at least two samples.")
    if not np.isfinite(matrix).all():
        raise ValueError("Internal clustering metrics require finite feature values.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Internal clustering metrics cannot normalize zero-length feature vectors.")
    return matrix / norms


def evaluate_internal_clustering(
    normalized_features,
    predictions: list[int],
    *,
    metrics: tuple[str, ...] | list[str] = INTERNAL_CLUSTERING_METRICS,
    silhouette_metric: str = "cosine",
    silhouette_sample_size: int | None = 5000,
    random_state: int | None = 42,
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    matrix = np.asarray(normalized_features, dtype=np.float32)
    prediction_array = np.asarray(predictions)
    if matrix.ndim != 2:
        raise ValueError(f"Internal clustering metrics require a 2D feature matrix, got shape {matrix.shape}.")
    if prediction_array.ndim != 1:
        raise ValueError(
            f"Internal clustering metrics require a 1D prediction array, got shape {prediction_array.shape}."
        )
    if matrix.shape[0] != prediction_array.shape[0]:
        raise ValueError(
            "Internal clustering metric inputs are misaligned: "
            f"{matrix.shape[0]} feature row(s) and {prediction_array.shape[0]} prediction(s)."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Internal clustering metrics require finite feature values.")

    n_samples = int(prediction_array.shape[0])
    n_clusters = int(np.unique(prediction_array).size)
    if n_clusters < 2 or n_clusters >= n_samples:
        raise ValueError(
            "Internal clustering metrics require 2 <= number of predicted clusters < number of samples, "
            f"got {n_clusters} cluster(s) and {n_samples} sample(s)."
        )
    if silhouette_sample_size is not None and silhouette_sample_size < 2:
        raise ValueError("silhouette_sample_size must be at least 2 or None.")

    effective_sample_size = (
        None
        if silhouette_sample_size is None or silhouette_sample_size >= n_samples
        else int(silhouette_sample_size)
    )
    results: dict[str, float] = {}
    for metric in metrics:
        if metric == "sil":
            results["sil"] = float(
                silhouette_score(
                    matrix,
                    prediction_array,
                    metric=silhouette_metric,
                    sample_size=effective_sample_size,
                    random_state=random_state,
                )
            )
        elif metric == "dbi":
            results["dbi"] = float(davies_bouldin_score(matrix, prediction_array))
        elif metric == "chi":
            results["chi"] = float(calinski_harabasz_score(matrix, prediction_array))
        else:
            raise ValueError(f"Unknown internal evaluation metric '{metric}'.")
    return results
