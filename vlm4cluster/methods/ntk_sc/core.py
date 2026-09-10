from __future__ import annotations

import csv
import hashlib
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.datasets import load_image_dataset
from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label
from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.feature_cache import (
    COMMON_WORDNET_PROMPT_BUILDERS,
    COMMON_WORDNET_PROMPT_TEMPLATE_KEY,
    cache_file_path,
    cache_key,
    common_feature_cache_dir,
    image_dataset_cache_key,
    load_or_compute_raw_image_embeddings,
    load_or_compute_text_prompt_bank,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
    wordnet_source_token,
)
from vlm4cluster.methods.tac.core import (
    _normalize_dataset_name,
    _load_tac_wordnet_nouns,
    _resolve_domain_shift_datasets,
    _resolve_tac_wordnet_csv,
)
from vlm4cluster.models import OpenCLIPBundle, load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import FAISS_KMEANS_CACHE_TOKEN, run_faiss_kmeans
from vlm4cluster.utils.io import ensure_dir
from vlm4cluster.utils.progress import get_progress_logger


DEFAULT_RED_DENSE_MAX_SAMPLES = 50_000
DEFAULT_RED_OUT_OF_CORE_MIN_SAMPLES = 20_000
DEFAULT_PNTK_BLOCK_SIZE = 512
DEFAULT_PNTK_COLUMN_BLOCK_SIZE = 1024
DEFAULT_RED_BLOCK_SIZE = 128
DEFAULT_MAX_SPARSE_NNZ = 150_000_000

NTK_SC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

NTK_SC_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10"): {"pntk_temperature": 0.05, "filter_cluster_num": 167},
    ("laion400m", "vitb32", "cifar20"): {"pntk_temperature": 0.01, "filter_cluster_num": 167},
    ("laion400m", "vitb32", "cifar100"): {"pntk_temperature": 0.05, "filter_cluster_num": 167},
    ("laion400m", "vitb32", "stl10"): {"pntk_temperature": 0.06},
    ("laion400m", "vitb32", "imagenet10"): {"pntk_temperature": 0.02, "filter_cluster_num": 43},
    ("laion400m", "vitb32", "imagenetdogs"): {"pntk_temperature": 0.01, "filter_cluster_num": 65},
    ("laion400m", "vitb32", "dtd"): {"pntk_temperature": 0.05},
    ("laion400m", "vitb32", "imagenet"): {"filter_cluster_num": 4271},
    ("laion400m", "vitb32", "places365standard"): {"filter_cluster_num": 6012},
    ("laion400m", "vitb32", "aircraft"): {"pntk_temperature": 0.07},
    ("laion400m", "vitb32", "cars"): {"pntk_temperature": 0.09, "filter_cluster_num": 300},
    ("laion400m", "vitb32", "flowers"): {"pntk_temperature": 0.03, "filter_cluster_num": 300},
    ("laion400m", "vitb32", "food"): {"pntk_temperature": 0.06, "filter_cluster_num": 300},
    ("laion400m", "vitb32", "pets"): {"filter_cluster_num": 100},
    ("laion400m", "vitb16", "cifar10"): {"pntk_temperature": 0.07, "filter_cluster_num": 167},
    ("laion400m", "vitb16", "cifar20"): {"pntk_temperature": 0.03, "filter_cluster_num": 167},
    ("laion400m", "vitb16", "cifar100"): {"filter_cluster_num": 167},
    ("laion400m", "vitb16", "stl10"): {"pntk_temperature": 0.06},
    ("laion400m", "vitb16", "imagenet10"): {"pntk_temperature": 0.02, "filter_cluster_num": 43},
    ("laion400m", "vitb16", "imagenetdogs"): {"filter_cluster_num": 65},
    ("laion400m", "vitb16", "dtd"): {"pntk_temperature": 0.06},
    ("laion400m", "vitb16", "imagenet"): {"filter_cluster_num": 4271},
    ("laion400m", "vitb16", "places365standard"): {"filter_cluster_num": 6012},
    ("laion400m", "vitb16", "aircraft"): {"pntk_temperature": 0.07},
    ("laion400m", "vitb16", "cars"): {"pntk_temperature": 0.09, "filter_cluster_num": 300},
    ("laion400m", "vitb16", "flowers"): {"pntk_temperature": 0.05, "filter_cluster_num": 300},
    ("laion400m", "vitb16", "food"): {"pntk_temperature": 0.06, "filter_cluster_num": 300},
    ("laion400m", "vitb16", "pets"): {"filter_cluster_num": 100},
    ("laion400m", "vitl14", "cifar10"): {"pntk_temperature": 0.07, "filter_cluster_num": 167},
    ("laion400m", "vitl14", "cifar20"): {"pntk_temperature": 0.05, "filter_cluster_num": 167},
    ("laion400m", "vitl14", "cifar100"): {"pntk_temperature": 0.05, "filter_cluster_num": 167},
    ("laion400m", "vitl14", "stl10"): {"pntk_temperature": 0.01},
    ("laion400m", "vitl14", "imagenet10"): {"pntk_temperature": 0.02, "filter_cluster_num": 43},
    ("laion400m", "vitl14", "imagenetdogs"): {"pntk_temperature": 0.01, "filter_cluster_num": 65},
    ("laion400m", "vitl14", "ucf101"): {"pntk_temperature": 0.05},
    ("laion400m", "vitl14", "imagenet"): {"filter_cluster_num": 4271},
    ("laion400m", "vitl14", "places365standard"): {"filter_cluster_num": 6012},
    ("laion400m", "vitl14", "aircraft"): {"pntk_temperature": 0.07},
    ("laion400m", "vitl14", "cars"): {"pntk_temperature": 0.09, "filter_cluster_num": 300},
    ("laion400m", "vitl14", "flowers"): {"pntk_temperature": 0.05, "filter_cluster_num": 300},
    ("laion400m", "vitl14", "food"): {"pntk_temperature": 0.06, "filter_cluster_num": 300},
    ("laion400m", "vitl14", "pets"): {"pntk_temperature": 0.05, "filter_cluster_num": 100},
}


@dataclass(slots=True)
class NTKSCOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class PreprocessedDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, dataset: LoadedImageDataset, preprocess) -> None:
        self.dataset = dataset
        self.preprocess = preprocess
        transforms = require_module("torchvision.transforms", "pip install torch torchvision")
        self.to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.dataset.sample_indices)

    def __getitem__(self, index: int):
        dataset_index = self.dataset.sample_indices[index]
        image, raw_label = self.dataset.dataset[dataset_index]
        if hasattr(image, "detach"):
            image = self.to_pil(image)
        image = self.preprocess(image)
        label = get_image_label(self.dataset, dataset_index, raw_label)
        return image, label, f"image-{self.dataset.split}-{dataset_index}"


def _normalize_ntk_sc_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_ntk_sc_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in NTK_SC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _resolve_ntk_sc_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_ntk_sc_profile_token(openclip_pretraining),
        _normalize_ntk_sc_profile_token(openclip_backbone),
        _normalize_ntk_sc_profile_dataset(dataset_name),
    )
    return dict(NTK_SC_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_ntk_sc_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    profile_key = (
        _normalize_ntk_sc_profile_token(openclip_pretraining),
        _normalize_ntk_sc_profile_token(openclip_backbone),
        _normalize_ntk_sc_profile_dataset(dataset_name),
    )
    if profile_key not in NTK_SC_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_ntk_sc_graph_k(params: dict[str, Any], defaults: dict[str, Any]) -> int:
    if "graph_k" in params:
        return int(params["graph_k"])
    if "k" in params:
        return int(params["k"])
    return int(defaults.get("graph_k", 30))


def _resolve_ntk_sc_pntk_temperature(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "pntk_temperature" in params:
        return float(params["pntk_temperature"])
    if "temp" in params:
        return float(params["temp"])
    return float(defaults.get("pntk_temperature", 0.04))


def _resolve_ntk_sc_filter_cluster_num(
    params: dict[str, Any],
    defaults: dict[str, Any],
    cluster_num: int,
) -> tuple[int, str]:
    if "filter_cluster_num" in params:
        return int(params["filter_cluster_num"]), "explicit"
    if "proxy_cluster_num" in params:
        return int(params["proxy_cluster_num"]), "explicit"
    if "filter_cluster_num" in defaults:
        return int(defaults["filter_cluster_num"]), "profile"
    return int(max(cluster_num * 3, cluster_num)), "inferred_default_3x_clusters"


def run_ntk_sc_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> NTKSCOutputs:
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger("ntk_sc", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "ntk_sc",
        bundle.spec.cache_key,
    )

    train_dataset, test_dataset, train_split, test_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="NTK-SC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={test_dataset.name}/{test_split}"
    )

    if train_dataset.num_classes == 0:
        raise ValueError("NTK-SC requires datasets with known class names.")

    cluster_num = int(params.get("n_clusters", test_dataset.num_classes))
    hyperparameter_defaults = _resolve_ntk_sc_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        test_dataset.name,
    )
    hyperparameter_profile_source = _resolve_ntk_sc_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        test_dataset.name,
    )
    filter_cluster_num, filter_cluster_num_source = _resolve_ntk_sc_filter_cluster_num(
        params,
        hyperparameter_defaults,
        cluster_num,
    )

    image_batch_size = int(params.get("image_batch_size", 1024))
    text_batch_size = int(params.get("text_batch_size", 2048))
    selected_nouns_per_cluster = int(params.get("selected_nouns_per_cluster", params.get("topK", 5)))
    graph_k = _resolve_ntk_sc_graph_k(params, hyperparameter_defaults)
    pntk_temperature = _resolve_ntk_sc_pntk_temperature(params, hyperparameter_defaults)
    mu = float(params.get("mu", 0.1))
    lam = float(params.get("lam", 10.0))
    t_outer = int(params.get("t_outer", 5))
    t_diffuse = int(params.get("t_diffuse", 8))
    ta_cd = int(params.get("ta_cd", 4))
    use_pntk_fast = bool(params.get("use_pntk_fast", True))
    red_backend_request = str(params.get("red_backend", params.get("red_mode", "gpu_auto")))
    red_oom_fallback = bool(params.get("red_oom_fallback", True))
    red_oom_fallback_backend = str(params.get("red_oom_fallback_backend", "dense_cached_gpu_red"))
    red_dense_max_samples = int(params.get("red_dense_max_samples", DEFAULT_RED_DENSE_MAX_SAMPLES))
    red_out_of_core_min_samples = int(
        params.get("red_out_of_core_min_samples", DEFAULT_RED_OUT_OF_CORE_MIN_SAMPLES)
    )
    pntk_block_size = int(params.get("pntk_block_size", DEFAULT_PNTK_BLOCK_SIZE))
    pntk_column_block_size = int(params.get("pntk_column_block_size", DEFAULT_PNTK_COLUMN_BLOCK_SIZE))
    red_block_size = int(params.get("red_block_size", DEFAULT_RED_BLOCK_SIZE))
    red_prune_k_param = params.get("red_prune_k")
    red_prune_k = None if red_prune_k_param is None else int(red_prune_k_param)
    max_sparse_nnz = int(params.get("max_sparse_nnz", DEFAULT_MAX_SPARSE_NNZ))
    spectral_n_init = int(params.get("spectral_n_init", params.get("n_init", 10)))
    force_recompute_red_graphs = bool(params.get("force_recompute_red_graphs", False))
    seed = int(params.get("seed", 0))

    if cluster_num <= 0:
        raise ValueError("NTK-SC requires n_clusters > 0.")
    if filter_cluster_num <= 0:
        raise ValueError("NTK-SC requires filter_cluster_num > 0.")
    if selected_nouns_per_cluster <= 0:
        raise ValueError("NTK-SC requires selected_nouns_per_cluster > 0.")
    if graph_k <= 0:
        raise ValueError("NTK-SC requires graph_k > 0.")
    if pntk_temperature <= 0:
        raise ValueError("NTK-SC requires pntk_temperature > 0.")
    if red_dense_max_samples <= 0:
        raise ValueError("NTK-SC requires red_dense_max_samples > 0.")
    if red_out_of_core_min_samples <= 0:
        raise ValueError("NTK-SC requires red_out_of_core_min_samples > 0.")
    if pntk_block_size <= 0:
        raise ValueError("NTK-SC requires pntk_block_size > 0.")
    if pntk_column_block_size <= 0:
        raise ValueError("NTK-SC requires pntk_column_block_size > 0.")
    if red_block_size <= 0:
        raise ValueError("NTK-SC requires red_block_size > 0.")
    if red_prune_k is not None and red_prune_k <= 0:
        raise ValueError("NTK-SC requires red_prune_k > 0 when it is provided.")
    if max_sparse_nnz <= 0:
        raise ValueError("NTK-SC requires max_sparse_nnz > 0.")
    if spectral_n_init <= 0:
        raise ValueError("NTK-SC requires spectral_n_init > 0.")

    with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
        train_image_features, train_labels = _encode_image_dataset(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    with progress.stage(f"Encoding evaluation image split '{test_dataset.name}/{test_split}'"):
        test_image_features, test_labels = _encode_image_dataset(
            dataset=test_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )

    wordnet_csv = _resolve_wordnet_csv(params)
    with progress.stage(f"Encoding WordNet noun prompt bank from {wordnet_csv}"):
        candidate_nouns, prompt_embeddings, ensemble_embeddings = _encode_wordnet_nouns(
            csv_path=wordnet_csv,
            bundle=bundle,
            batch_size=text_batch_size,
            device=device,
            cache_dir=output_dir,
            force_recompute=bool(params.get("force_recompute_nouns", False)),
        )
    efficiency_measurement = start_train_eval_measurement(device)
    with progress.stage(f"Selecting NTK-SC nouns with {filter_cluster_num} proxy cluster(s)"):
        selected_indices, selected_nouns, selected_prompt_embeddings = _select_discriminative_nouns(
            dataset_name=image_dataset_cache_key(train_dataset),
            dataset_split=None,
            wordnet_csv=wordnet_csv,
            image_features=train_image_features,
            candidate_nouns=candidate_nouns,
            prompt_embeddings=prompt_embeddings,
            ensemble_embeddings=ensemble_embeddings,
            filter_cluster_num=filter_cluster_num,
            selected_nouns_per_cluster=selected_nouns_per_cluster,
            bundle=bundle,
            cache_dir=output_dir,
            force_recompute=bool(params.get("force_recompute_selected_nouns", False)),
            seed=seed,
            device=device,
        )

    selected_indices_hash = _array_digest(selected_indices)
    red_graph_cache_key = cache_key(
        image_dataset_cache_key(test_dataset),
        f"n{test_image_features.shape[0]}",
        bundle.spec.cache_key,
        wordnet_source_token(wordnet_csv),
        COMMON_WORDNET_PROMPT_TEMPLATE_KEY,
        f"source{train_dataset.name}",
        f"filter{filter_cluster_num}",
        f"top{selected_nouns_per_cluster}",
        f"sel{selected_indices_hash}",
        f"graph{graph_k}",
        f"temp{pntk_temperature}",
    )

    with progress.stage(f"Running RED spectral clustering with {cluster_num} cluster(s)"):
        predictions, red_metadata = _run_red_spectral_clustering(
            image_features=test_image_features,
            prompt_embeddings=selected_prompt_embeddings,
            cluster_num=cluster_num,
            graph_k=graph_k,
            pntk_temperature=pntk_temperature,
            mu=mu,
            lam=lam,
            t_outer=t_outer,
            t_diffuse=t_diffuse,
            ta_cd=ta_cd,
            seed=seed,
            use_pntk_fast=use_pntk_fast,
            device=device,
            red_backend_request=red_backend_request,
            red_oom_fallback=red_oom_fallback,
            red_oom_fallback_backend=red_oom_fallback_backend,
            red_dense_max_samples=red_dense_max_samples,
            red_out_of_core_min_samples=red_out_of_core_min_samples,
            pntk_block_size=pntk_block_size,
            pntk_column_block_size=pntk_column_block_size,
            red_block_size=red_block_size,
            red_prune_k=red_prune_k,
            max_sparse_nnz=max_sparse_nnz,
            spectral_n_init=spectral_n_init,
            cache_dir=output_dir,
            red_graph_cache_key=red_graph_cache_key,
            force_recompute_red_graphs=force_recompute_red_graphs,
            keep_red_work_dir=bool(params.get("keep_red_work_dir", False)),
            red_work_dir=params.get("red_work_dir"),
            progress=progress,
        )
    efficiency_measurement.stop()

    return NTKSCOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=test_labels.tolist() if test_labels is not None else None,
        evaluation_split=test_split,
        metadata={
            "dataset": train_dataset.name,
            "source_dataset": train_dataset.name,
            "evaluation_dataset": test_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": test_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "wordnet_csv": str(wordnet_csv),
            "cluster_num": cluster_num,
            "filter_cluster_num": filter_cluster_num,
            "filter_cluster_num_source": filter_cluster_num_source,
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "selected_nouns_per_cluster": selected_nouns_per_cluster,
            "selected_noun_count": len(selected_nouns),
            "selected_noun_examples": selected_nouns[:10],
            "selected_noun_indices_preview": selected_indices[:10].tolist(),
            "red_oom_fallback": red_oom_fallback,
            "red_oom_fallback_backend": red_oom_fallback_backend,
            "red_out_of_core_min_samples": red_out_of_core_min_samples,
            "train_image_count": int(train_image_features.shape[0]),
            "test_image_count": int(test_image_features.shape[0]),
            "train_labels_available": train_labels is not None,
            **red_metadata,
        },
    )


def _build_split_dataset(inputs: MethodInputs, split: str, max_samples: Any) -> LoadedImageDataset:
    config = replace(
        inputs.image_dataset_config,
        split=split,
        transform_preset="none",
        max_samples=max_samples if max_samples is not None else inputs.image_dataset_config.max_samples,
    )
    return load_image_dataset(config)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def _encode_image_dataset(
    dataset: LoadedImageDataset,
    bundle: OpenCLIPBundle,
    batch_size: int,
    num_workers: int,
    device,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    del cache_dir
    raw_features, labels = load_or_compute_raw_image_embeddings(
        dataset=dataset,
        bundle=bundle,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    return _normalize_rows(raw_features.astype("float32")), labels


def _resolve_wordnet_csv(params: dict[str, Any]) -> Path:
    return _resolve_tac_wordnet_csv(params)


def _load_wordnet_nouns(csv_path: Path) -> list[str]:
    return _load_tac_wordnet_nouns(csv_path, None)


def _encode_wordnet_nouns(
    csv_path: Path,
    bundle: OpenCLIPBundle,
    batch_size: int,
    device,
    cache_dir: Path,
    force_recompute: bool,
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    del cache_dir
    nouns = _load_wordnet_nouns(csv_path)
    _, prompt_embeddings, ensemble_embeddings = load_or_compute_text_prompt_bank(
        csv_path=csv_path,
        nouns=nouns,
        bundle=bundle,
        batch_size=batch_size,
        device=device,
        cache_dir=common_feature_cache_dir(csv_path.parent, "text"),
        prompt_builders=COMMON_WORDNET_PROMPT_BUILDERS,
        template_key=COMMON_WORDNET_PROMPT_TEMPLATE_KEY,
        normalize_prompt_embeddings=True,
        normalize_ensemble=True,
        force_recompute=force_recompute,
    )
    return nouns, prompt_embeddings, ensemble_embeddings


def _select_discriminative_nouns(
    dataset_name: str,
    dataset_split: str | None,
    wordnet_csv: Path,
    image_features: np.ndarray,
    candidate_nouns: list[str],
    prompt_embeddings: list[np.ndarray],
    ensemble_embeddings: np.ndarray,
    filter_cluster_num: int,
    selected_nouns_per_cluster: int,
    bundle: OpenCLIPBundle,
    cache_dir: Path,
    force_recompute: bool,
    seed: int,
    device,
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    selection_cache_key = cache_key(
        dataset_name,
        dataset_split,
        bundle.spec.cache_key,
        wordnet_source_token(wordnet_csv),
        COMMON_WORDNET_PROMPT_TEMPLATE_KEY,
        FAISS_KMEANS_CACHE_TOKEN,
        f"filter{filter_cluster_num}",
        f"top{selected_nouns_per_cluster}",
        f"seed{seed}",
    )
    cache_path = cache_file_path(cache_dir, f"{selection_cache_key}__selected_nouns", ".npz")
    if cache_path.exists() and not force_recompute:
        payload = np.load(cache_path, allow_pickle=True)
        selected_indices = payload["selected_indices"].astype(np.int64)
        selected_nouns = [str(noun) for noun in payload["selected_nouns"].tolist()]
        stacked = payload["selected_prompt_embeddings"]
        return selected_indices, selected_nouns, [stacked[index] for index in range(stacked.shape[0])]

    torch = require_module("torch", "pip install torch")
    normalized_images = _normalize_rows(image_features.astype("float32"))
    normalized_nouns = _normalize_rows(ensemble_embeddings.astype("float32"))
    preds = _run_kmeans(
        normalized_images,
        cluster_num=filter_cluster_num,
        n_iter=300,
        n_redo=10,
        random_state=seed,
    )

    image_centers = np.zeros((filter_cluster_num, normalized_images.shape[1]), dtype=np.float32)
    for index in range(filter_cluster_num):
        members = normalized_images[preds == index]
        if len(members) == 0:
            continue
        image_centers[index] = members.mean(axis=0)
    image_centers = _normalize_rows(image_centers)

    softmax_nouns = _compute_center_noun_softmax(image_centers, normalized_nouns, device)
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
    for index in range(filter_cluster_num):
        if int((class_pred == index).sum()) == 0:
            continue
        class_index = torch.where(class_pred == index)[0]
        softmax_class = softmax_nouns[:, class_index]
        confidence = softmax_class.max(dim=0)[0]
        rank = torch.argsort(confidence, descending=True)
        selected_idx[class_index[rank[:selected_nouns_per_cluster]]] = True

    selected_indices = np.where(selected_idx.cpu().numpy())[0].astype(np.int64)
    selected_nouns = [candidate_nouns[index] for index in selected_indices.tolist()]
    stacked = np.stack([prompt_embedding[selected_indices] for prompt_embedding in prompt_embeddings], axis=0)
    np.savez_compressed(
        cache_path,
        selected_indices=selected_indices,
        selected_nouns=np.asarray(selected_nouns, dtype=object),
        selected_prompt_embeddings=stacked,
    )
    return selected_indices, selected_nouns, [stacked[index] for index in range(stacked.shape[0])]


def _compute_center_noun_softmax(
    image_centers: np.ndarray,
    normalized_nouns: np.ndarray,
    device,
):
    torch = require_module("torch", "pip install torch")
    torch_device = torch.device(device)
    centers = nouns = similarity = None
    try:
        with torch.no_grad():
            centers = torch.from_numpy(image_centers).to(torch_device).float()
            nouns = torch.from_numpy(normalized_nouns).to(torch_device).float()
            similarity = centers @ nouns.t()
            softmax_nouns = torch.softmax(similarity, dim=0).detach().cpu().float()
    except RuntimeError as exc:
        if torch_device.type != "cuda" or not _is_oom_error(exc):
            raise
        centers = nouns = similarity = None
        _clear_torch_cuda_cache()
        with torch.no_grad():
            centers = torch.from_numpy(image_centers).float()
            nouns = torch.from_numpy(normalized_nouns).float()
            similarity = centers @ nouns.t()
            softmax_nouns = torch.softmax(similarity, dim=0).detach().cpu().float()
    finally:
        del centers, nouns, similarity
        if torch_device.type == "cuda":
            _clear_torch_cuda_cache()
    return softmax_nouns


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha1(contiguous.view(np.uint8)).hexdigest()[:12]


def _run_kmeans(
    features: np.ndarray,
    cluster_num: int,
    n_iter: int,
    n_redo: int,
    random_state: int | None,
) -> np.ndarray:
    assignments, _ = run_faiss_kmeans(
        features.astype("float32"),
        n_clusters=cluster_num,
        n_iter=n_iter,
        n_redo=n_redo,
        spherical=True,
        random_state=random_state,
    )
    return assignments


def _build_knn_graph_from_similarity(similarity: np.ndarray, k: int) -> np.ndarray:
    node_count = similarity.shape[0]
    if node_count <= 1:
        return np.zeros_like(similarity, dtype=np.float32)

    graph = similarity.copy()
    np.fill_diagonal(graph, 0.0)
    effective_k = min(k, node_count - 1)
    indices = np.argpartition(-graph, kth=effective_k - 1, axis=1)[:, :effective_k]
    rows = np.repeat(np.arange(node_count), indices.shape[1])
    cols = indices.reshape(-1)
    values = graph[rows, cols]
    adjacency = np.zeros_like(graph, dtype=np.float32)
    adjacency[rows, cols] = values
    adjacency = np.maximum(adjacency, adjacency.T)
    np.nan_to_num(adjacency, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = np.maximum(adjacency, 0.0)
    return adjacency


def _transition_from_w(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    zero_mask = degree <= 0
    if zero_mask.any():
        adjacency = adjacency.copy()
        adjacency[zero_mask, :] = 0.0
        adjacency[:, zero_mask] = 0.0
        degree = adjacency.sum(axis=1)
    d_m12 = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    transition = (d_m12[:, None] * adjacency) * d_m12[None, :]
    transition[zero_mask, :] = 0.0
    transition[:, zero_mask] = 0.0
    return transition


def _compute_pntk_matrix_fast(images_emb, nouns_emb, temp: float, kx_precomputed=None) -> np.ndarray:
    if kx_precomputed is None:
        kx = images_emb @ images_emb.t()
    else:
        kx = kx_precomputed
    logits = (images_emb @ nouns_emb.t()) / temp
    soft_assignments = __import__("torch").softmax(logits, dim=1)
    ks = soft_assignments @ soft_assignments.t()
    kernel = (kx * ks) / (temp * temp)
    return kernel.detach().cpu().float().numpy()


def _compute_pntk_matrix_autograd(images_emb, nouns_emb, temp: float) -> np.ndarray:
    torch = require_module("torch", "pip install torch")
    sample_count = images_emb.shape[0]
    grads = []
    for index in range(sample_count):
        if nouns_emb.grad is not None:
            nouns_emb.grad.zero_()
        image_vector = images_emb[index : index + 1]
        logits = (image_vector @ nouns_emb.t()) / temp
        scalar = torch.logsumexp(logits, dim=1).sum()
        grad = torch.autograd.grad(scalar, nouns_emb, create_graph=False, retain_graph=False)[0]
        grads.append(grad.flatten().detach().cpu())
    gradients = torch.stack(grads, dim=0)
    return (gradients @ gradients.t()).numpy().astype(np.float32)


def _compute_hv(adjacency: np.ndarray, transition: np.ndarray) -> float:
    diffusion = transition @ adjacency @ transition.T
    return float(np.sum(adjacency * adjacency) - np.sum(adjacency * diffusion))


def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
    vector = np.maximum(vector, 0.0)
    if float(vector.sum()) == 0.0:
        return np.ones_like(vector) / len(vector)
    sorted_vector = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_vector)
    cond = sorted_vector > (cumulative - 1) / (np.arange(len(vector)) + 1)
    if not cond.any():
        return np.ones_like(vector) / len(vector)
    rho = np.where(cond)[0][-1]
    theta = (cumulative[rho] - 1) / (rho + 1.0)
    return np.maximum(vector - theta, 0.0)


def _update_beta_coordinate_descent(beta: np.ndarray, hv_values: np.ndarray, lam: float, passes: int) -> np.ndarray:
    count = len(beta)
    for _ in range(passes):
        for i in range(count):
            for j in range(i + 1, count):
                beta_sum = beta[i] + beta[j]
                numerator = lam * beta_sum + (hv_values[j] - hv_values[i])
                beta_i_new = numerator / (2.0 * lam)
                beta_i_new = min(max(beta_i_new, 0.0), 1.0)
                beta_j_new = beta_sum - beta_i_new
                beta_j_new = min(max(beta_j_new, 0.0), 1.0)
                beta[i], beta[j] = beta_i_new, beta_j_new
        beta = _project_to_simplex(beta)
    return beta


def _run_red_spectral_clustering(
    image_features: np.ndarray,
    prompt_embeddings: list[np.ndarray],
    cluster_num: int,
    graph_k: int,
    pntk_temperature: float,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    use_pntk_fast: bool,
    device,
    red_backend_request: str,
    red_oom_fallback: bool,
    red_oom_fallback_backend: str,
    red_dense_max_samples: int,
    red_out_of_core_min_samples: int,
    pntk_block_size: int,
    pntk_column_block_size: int,
    red_block_size: int,
    red_prune_k: int | None,
    max_sparse_nnz: int,
    spectral_n_init: int,
    cache_dir: Path,
    red_graph_cache_key: str,
    force_recompute_red_graphs: bool,
    keep_red_work_dir: bool,
    red_work_dir: Any,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_count = int(image_features.shape[0])
    if cluster_num >= sample_count:
        raise ValueError(
            f"NTK-SC requires n_clusters < number of evaluation samples, got {cluster_num} and {sample_count}."
        )

    red_backend = _resolve_red_backend(
        red_backend_request,
        sample_count=sample_count,
        dense_max_samples=red_dense_max_samples,
        out_of_core_min_samples=red_out_of_core_min_samples,
        device=device,
    )
    if red_backend not in {"dense", "dense_gpu"} and not use_pntk_fast:
        raise ValueError("NTK-SC non-dense RED backends require use_pntk_fast=True.")

    if red_backend == "dense":
        try:
            predictions, metadata = _run_red_spectral_clustering_dense(
                image_features=image_features,
                prompt_embeddings=prompt_embeddings,
                cluster_num=cluster_num,
                graph_k=graph_k,
                pntk_temperature=pntk_temperature,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                use_pntk_fast=use_pntk_fast,
                device=device,
                progress=progress,
            )
            fallback_triggered = False
            fallback_error = None
        except (MemoryError, RuntimeError) as exc:
            if not red_oom_fallback or not _is_oom_error(exc) or not use_pntk_fast:
                raise
            fallback_backend = _resolve_red_fallback_backend(red_oom_fallback_backend)
            _clear_torch_cuda_cache()
            if progress is not None:
                progress.log(
                    "Original dense RED raised an OOM error; "
                    f"falling back to NTK-SC red_backend='{fallback_backend}'."
                )
            predictions, metadata = _run_red_cached_fallback(
                fallback_backend=fallback_backend,
                image_features=image_features,
                prompt_embeddings=prompt_embeddings,
                cluster_num=cluster_num,
                graph_k=graph_k,
                pntk_temperature=pntk_temperature,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                device=device,
                pntk_block_size=pntk_block_size,
                pntk_column_block_size=pntk_column_block_size,
                spectral_n_init=spectral_n_init,
                red_block_size=red_block_size,
                red_prune_k=red_prune_k,
                max_sparse_nnz=max_sparse_nnz,
                cache_dir=cache_dir,
                red_graph_cache_key=red_graph_cache_key,
                force_recompute_red_graphs=force_recompute_red_graphs,
                keep_red_work_dir=keep_red_work_dir,
                red_work_dir=red_work_dir,
                progress=progress,
            )
            fallback_triggered = True
            fallback_error = str(exc)
        metadata.update(
            {
                "red_backend": metadata.get("red_backend", red_backend),
                "red_backend_request": red_backend_request,
                "red_oom_fallback_triggered": fallback_triggered,
                "red_oom_fallback_error": fallback_error,
                "red_dense_max_samples": red_dense_max_samples,
                "red_out_of_core_min_samples": red_out_of_core_min_samples,
                "pntk_block_size": metadata.get("pntk_block_size"),
                "pntk_column_block_size": metadata.get("pntk_column_block_size"),
                "red_block_size": metadata.get("red_block_size"),
                "red_prune_k": metadata.get("red_prune_k"),
                "max_sparse_nnz": metadata.get("max_sparse_nnz"),
                "spectral_n_init": metadata.get("spectral_n_init"),
                "red_graph_cache_key": metadata.get("red_graph_cache_key"),
                "red_transition_cache_paths": metadata.get("red_transition_cache_paths", []),
                "red_transition_cache_hits": metadata.get("red_transition_cache_hits", []),
            }
        )
        return predictions, metadata

    if red_backend == "dense_gpu":
        try:
            predictions, metadata = _run_red_spectral_clustering_dense_gpu(
                image_features=image_features,
                prompt_embeddings=prompt_embeddings,
                cluster_num=cluster_num,
                graph_k=graph_k,
                pntk_temperature=pntk_temperature,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                use_pntk_fast=use_pntk_fast,
                device=device,
                progress=progress,
            )
            fallback_triggered = False
            fallback_error = None
        except (MemoryError, RuntimeError) as exc:
            if not red_oom_fallback or not _is_oom_error(exc) or not use_pntk_fast:
                raise
            fallback_backend = _resolve_red_fallback_backend(red_oom_fallback_backend)
            _clear_torch_cuda_cache()
            if progress is not None:
                progress.log(
                    "GPU dense RED raised an OOM error; "
                    f"falling back to NTK-SC red_backend='{fallback_backend}'."
                )
            predictions, metadata = _run_red_cached_fallback(
                fallback_backend=fallback_backend,
                image_features=image_features,
                prompt_embeddings=prompt_embeddings,
                cluster_num=cluster_num,
                graph_k=graph_k,
                pntk_temperature=pntk_temperature,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                device=device,
                pntk_block_size=pntk_block_size,
                pntk_column_block_size=pntk_column_block_size,
                spectral_n_init=spectral_n_init,
                red_block_size=red_block_size,
                red_prune_k=red_prune_k,
                max_sparse_nnz=max_sparse_nnz,
                cache_dir=cache_dir,
                red_graph_cache_key=red_graph_cache_key,
                force_recompute_red_graphs=force_recompute_red_graphs,
                keep_red_work_dir=keep_red_work_dir,
                red_work_dir=red_work_dir,
                progress=progress,
            )
            fallback_triggered = True
            fallback_error = str(exc)
        metadata.update(
            {
                "red_backend": metadata.get("red_backend", "dense_gpu"),
                "red_backend_request": red_backend_request,
                "red_oom_fallback_triggered": fallback_triggered,
                "red_oom_fallback_error": fallback_error,
                "red_dense_max_samples": red_dense_max_samples,
                "red_out_of_core_min_samples": red_out_of_core_min_samples,
                "pntk_block_size": metadata.get("pntk_block_size"),
                "pntk_column_block_size": metadata.get("pntk_column_block_size"),
                "red_block_size": metadata.get("red_block_size"),
                "red_prune_k": metadata.get("red_prune_k"),
                "max_sparse_nnz": metadata.get("max_sparse_nnz"),
                "spectral_n_init": metadata.get("spectral_n_init"),
                "red_graph_cache_key": metadata.get("red_graph_cache_key"),
                "red_transition_cache_paths": metadata.get("red_transition_cache_paths", []),
                "red_transition_cache_hits": metadata.get("red_transition_cache_hits", []),
            }
        )
        return predictions, metadata

    if red_backend in {"dense_cached", "dense_cached_gpu_red"}:
        predictions, metadata = _run_red_cached_fallback(
            fallback_backend=red_backend,
            image_features=image_features,
            prompt_embeddings=prompt_embeddings,
            cluster_num=cluster_num,
            graph_k=graph_k,
            pntk_temperature=pntk_temperature,
            mu=mu,
            lam=lam,
            t_outer=t_outer,
            t_diffuse=t_diffuse,
            ta_cd=ta_cd,
            seed=seed,
            device=device,
            pntk_block_size=pntk_block_size,
            pntk_column_block_size=pntk_column_block_size,
            spectral_n_init=spectral_n_init,
            red_block_size=red_block_size,
            red_prune_k=red_prune_k,
            max_sparse_nnz=max_sparse_nnz,
            cache_dir=cache_dir,
            red_graph_cache_key=red_graph_cache_key,
            force_recompute_red_graphs=force_recompute_red_graphs,
            keep_red_work_dir=keep_red_work_dir,
            red_work_dir=red_work_dir,
            progress=progress,
        )
        metadata.update(
            {
                "red_backend": red_backend,
                "red_backend_request": red_backend_request,
                "red_oom_fallback_triggered": False,
                "red_oom_fallback_error": None,
                "red_dense_max_samples": red_dense_max_samples,
                "red_out_of_core_min_samples": red_out_of_core_min_samples,
                "red_prune_k": None,
                "max_sparse_nnz": None,
            }
        )
        return predictions, metadata

    transition_list, transition_cache_paths, transition_cache_hits = _load_or_build_sparse_pntk_transition_graphs(
        image_features=image_features,
        prompt_embeddings=prompt_embeddings,
        graph_k=graph_k,
        pntk_temperature=pntk_temperature,
        pntk_block_size=pntk_block_size,
        pntk_column_block_size=pntk_column_block_size,
        device=device,
        cache_dir=ensure_dir(cache_dir / "red_graphs"),
        red_graph_cache_key=red_graph_cache_key,
        force_recompute=force_recompute_red_graphs,
        progress=progress,
    )

    if red_backend == "full_sparse":
        predictions, metadata = _run_red_sparse(
            transition_list=transition_list,
            cluster_num=cluster_num,
            mu=mu,
            lam=lam,
            t_outer=t_outer,
            t_diffuse=t_diffuse,
            ta_cd=ta_cd,
            seed=seed,
            red_prune_k=red_prune_k,
            max_sparse_nnz=max_sparse_nnz,
            spectral_n_init=spectral_n_init,
            progress=progress,
        )
    elif red_backend == "out_of_core":
        predictions, metadata = _run_red_out_of_core(
            transition_list=transition_list,
            cluster_num=cluster_num,
            mu=mu,
            lam=lam,
            t_outer=t_outer,
            t_diffuse=t_diffuse,
            ta_cd=ta_cd,
            seed=seed,
            spectral_n_init=spectral_n_init,
            red_block_size=red_block_size,
            cache_dir=cache_dir,
            red_graph_cache_key=red_graph_cache_key,
            keep_red_work_dir=keep_red_work_dir,
            red_work_dir=red_work_dir,
            progress=progress,
        )
    else:
        raise ValueError(f"Unsupported NTK-SC red_backend '{red_backend}'.")

    metadata.update(
        {
            "graph_k": graph_k,
            "pntk_temperature": pntk_temperature,
            "red_backend": red_backend,
            "red_backend_request": red_backend_request,
            "red_oom_fallback_triggered": False,
            "red_oom_fallback_error": None,
            "red_dense_max_samples": red_dense_max_samples,
            "red_out_of_core_min_samples": red_out_of_core_min_samples,
            "pntk_block_size": pntk_block_size,
            "pntk_column_block_size": pntk_column_block_size,
            "red_block_size": red_block_size,
            "red_prune_k": red_prune_k,
            "max_sparse_nnz": max_sparse_nnz,
            "spectral_n_init": spectral_n_init,
            "red_graph_cache_key": red_graph_cache_key,
            "red_transition_cache_paths": [str(path) for path in transition_cache_paths],
            "red_transition_cache_hits": transition_cache_hits,
        }
    )
    return predictions, metadata


def _resolve_red_backend(
    requested: str,
    *,
    sample_count: int,
    dense_max_samples: int,
    out_of_core_min_samples: int,
    device=None,
) -> str:
    normalized = requested.strip().lower().replace("-", "_")
    aliases = {
        "gpu_auto": "gpu_auto",
        "auto_gpu": "gpu_auto",
        "gpu": "gpu_auto",
        "default": "gpu_auto",
        "auto": "auto",
        "dense": "dense",
        "dense_gpu": "dense_gpu",
        "densegpu": "dense_gpu",
        "gpu_dense": "dense_gpu",
        "dense_cached": "dense_cached",
        "densecached": "dense_cached",
        "gpu_cached": "dense_cached",
        "dense_cached_gpu_red": "dense_cached_gpu_red",
        "densecachedgpured": "dense_cached_gpu_red",
        "gpu_red_cached": "dense_cached_gpu_red",
        "sparse": "full_sparse",
        "full_sparse": "full_sparse",
        "out_of_core": "out_of_core",
        "outofcore": "out_of_core",
    }
    if normalized not in aliases:
        raise ValueError(
            "NTK-SC red_backend must be one of: gpu_auto, auto_gpu, auto, dense, dense_gpu, "
            "dense_cached, dense_cached_gpu_red, full_sparse, sparse, out_of_core."
        )
    resolved = aliases[normalized]
    if resolved == "gpu_auto":
        if _device_type_name(device) == "cuda":
            if sample_count <= dense_max_samples:
                return "dense_gpu"
            return "dense_cached_gpu_red"
        resolved = "auto"
    if resolved == "auto":
        if sample_count <= dense_max_samples:
            return "dense"
        if sample_count >= out_of_core_min_samples:
            return "out_of_core"
        return "full_sparse"
    return resolved


def _device_type_name(device) -> str:
    if device is None:
        return "cpu"
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type).lower()
    return str(device).split(":", maxsplit=1)[0].lower()


def _resolve_red_fallback_backend(requested: str) -> str:
    normalized = requested.strip().lower().replace("-", "_")
    aliases = {
        "dense_cached": "dense_cached",
        "densecached": "dense_cached",
        "gpu_cached": "dense_cached",
        "dense_cached_gpu_red": "dense_cached_gpu_red",
        "densecachedgpured": "dense_cached_gpu_red",
        "gpu_red_cached": "dense_cached_gpu_red",
        "out_of_core": "out_of_core",
        "outofcore": "out_of_core",
        "full_sparse": "full_sparse",
        "sparse": "full_sparse",
    }
    if normalized not in aliases:
        raise ValueError(
            "NTK-SC red_oom_fallback_backend must be one of: dense_cached_gpu_red, dense_cached, out_of_core, full_sparse."
        )
    return aliases[normalized]


def _is_oom_error(error: BaseException) -> bool:
    if isinstance(error, MemoryError):
        return True
    text = str(error).lower()
    name = type(error).__name__.lower()
    return "outofmemory" in name or "out of memory" in text or "cuda oom" in text


def _clear_torch_cuda_cache() -> None:
    try:
        torch = require_module("torch", "pip install torch")
    except RuntimeError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_red_spectral_clustering_dense(
    image_features: np.ndarray,
    prompt_embeddings: list[np.ndarray],
    cluster_num: int,
    graph_k: int,
    pntk_temperature: float,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    use_pntk_fast: bool,
    device,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = require_module("torch", "pip install torch")
    sklearn_cluster = require_module("sklearn.cluster", "pip install scikit-learn")
    progress = progress or get_progress_logger("ntk_sc")

    images_emb = torch.from_numpy(_normalize_rows(image_features.astype("float32"))).to(device).float()
    kx = images_emb @ images_emb.t() if use_pntk_fast else None

    transition_list = []
    for prompt_index, prompt_embedding in enumerate(prompt_embeddings, start=1):
        nouns_emb = torch.from_numpy(_normalize_rows(prompt_embedding.astype("float32"))).to(device).float()
        nouns_emb.requires_grad_(not use_pntk_fast)
        if use_pntk_fast:
            kernel = _compute_pntk_matrix_fast(images_emb, nouns_emb, pntk_temperature, kx_precomputed=kx)
        else:
            kernel = _compute_pntk_matrix_autograd(images_emb, nouns_emb, pntk_temperature)
        adjacency = _build_knn_graph_from_similarity(kernel, k=graph_k)
        transition_list.append(_transition_from_w(adjacency).astype(np.float32))
        del nouns_emb, kernel, adjacency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.step("Building pNTK transition graphs", prompt_index, len(prompt_embeddings), noun="prompt")

    sample_count = images_emb.shape[0]
    adjacency = np.eye(sample_count, dtype=np.float32)
    beta = np.ones(len(prompt_embeddings), dtype=np.float32) / len(prompt_embeddings)
    diffusion_deltas: list[float] = []

    for outer_index in range(t_outer):
        alpha = beta / (mu + float(beta.sum()))
        alpha_sum = float(alpha.sum())
        identity = np.eye(sample_count, dtype=np.float32)
        previous = adjacency
        for _ in range(t_diffuse):
            aggregated = np.zeros_like(adjacency)
            for index, transition in enumerate(transition_list):
                aggregated += alpha[index] * (transition @ adjacency @ transition.T)
            adjacency = aggregated + (1.0 - alpha_sum) * identity
            adjacency = 0.5 * (adjacency + adjacency.T)
            adjacency[adjacency < 0] = 0.0
            numerator = np.linalg.norm(adjacency - previous, "fro")
            denominator = np.linalg.norm(previous, "fro") + 1e-12
            delta = float(numerator / denominator)
            diffusion_deltas.append(delta)
            if delta < 1e-4:
                break
            previous = adjacency

        hv_values = np.asarray([_compute_hv(adjacency, transition) for transition in transition_list], dtype=np.float32)
        beta = _update_beta_coordinate_descent(beta, hv_values, lam, passes=ta_cd)
        progress.step("RED diffusion", outer_index + 1, t_outer, noun="outer step")

    spectral = sklearn_cluster.SpectralClustering(
        n_clusters=cluster_num,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=seed,
    )
    predictions = spectral.fit_predict(adjacency)
    return predictions.astype(np.int64), {
        "graph_k": graph_k,
        "pntk_temperature": pntk_temperature,
        "mu": mu,
        "lam": lam,
        "t_outer": t_outer,
        "t_diffuse": t_diffuse,
        "ta_cd": ta_cd,
        "use_pntk_fast": use_pntk_fast,
        "num_prompts": len(prompt_embeddings),
        "final_beta": beta.tolist(),
        "diffusion_delta_tail": diffusion_deltas[-10:],
        "final_affinity_nnz": int(np.count_nonzero(adjacency)),
    }


def _run_red_spectral_clustering_dense_gpu(
    image_features: np.ndarray,
    prompt_embeddings: list[np.ndarray],
    cluster_num: int,
    graph_k: int,
    pntk_temperature: float,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    use_pntk_fast: bool,
    device,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = require_module("torch", "pip install torch")
    sklearn_cluster = require_module("sklearn.cluster", "pip install scikit-learn")
    progress = progress or get_progress_logger("ntk_sc")
    device = _require_cuda_device(device, "dense_gpu")

    images_emb = torch.from_numpy(_normalize_rows(image_features.astype("float32"))).to(device).float()
    kx = images_emb @ images_emb.t() if use_pntk_fast else None

    transition_list = []
    for prompt_index, prompt_embedding in enumerate(prompt_embeddings, start=1):
        nouns_emb = torch.from_numpy(_normalize_rows(prompt_embedding.astype("float32"))).to(device).float()
        nouns_emb.requires_grad_(not use_pntk_fast)
        if use_pntk_fast:
            kernel = _compute_pntk_matrix_fast(images_emb, nouns_emb, pntk_temperature, kx_precomputed=kx)
        else:
            kernel = _compute_pntk_matrix_autograd(images_emb, nouns_emb, pntk_temperature)
        adjacency = _build_knn_graph_from_similarity(kernel, k=graph_k)
        transition = _transition_from_w(adjacency).astype(np.float32)
        transition_list.append(torch.from_numpy(transition).to(device).float())
        del nouns_emb, kernel, adjacency, transition
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.step("Building GPU pNTK transition graphs", prompt_index, len(prompt_embeddings), noun="prompt")

    sample_count = images_emb.shape[0]
    adjacency = torch.eye(sample_count, device=device, dtype=torch.float32)
    identity = torch.eye(sample_count, device=device, dtype=torch.float32)
    beta = np.ones(len(prompt_embeddings), dtype=np.float32) / len(prompt_embeddings)
    diffusion_deltas: list[float] = []

    with torch.no_grad():
        for outer_index in range(t_outer):
            alpha = beta / (mu + float(beta.sum()))
            alpha_sum = float(alpha.sum())
            previous = adjacency
            for _ in range(t_diffuse):
                aggregated = torch.zeros_like(adjacency)
                for index, transition in enumerate(transition_list):
                    aggregated.add_(transition @ adjacency @ transition.t(), alpha=float(alpha[index]))
                adjacency = aggregated + (1.0 - alpha_sum) * identity
                adjacency = 0.5 * (adjacency + adjacency.t())
                adjacency.clamp_(min=0.0)
                numerator = torch.linalg.matrix_norm(adjacency - previous)
                denominator = torch.linalg.matrix_norm(previous) + 1e-12
                delta = float((numerator / denominator).detach().cpu())
                diffusion_deltas.append(delta)
                if delta < 1e-4:
                    break
                previous = adjacency

            adjacency_norm_sq = torch.sum(adjacency * adjacency)
            hv_values = []
            for transition in transition_list:
                diffusion = transition @ adjacency @ transition.t()
                hv_values.append(float((adjacency_norm_sq - torch.sum(adjacency * diffusion)).detach().cpu()))
            beta = _update_beta_coordinate_descent(beta, np.asarray(hv_values, dtype=np.float32), lam, passes=ta_cd)
            progress.step("GPU RED diffusion", outer_index + 1, t_outer, noun="outer step")

    affinity = adjacency.detach().cpu().float().numpy()
    spectral = sklearn_cluster.SpectralClustering(
        n_clusters=cluster_num,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=seed,
    )
    predictions = spectral.fit_predict(affinity)
    return predictions.astype(np.int64), {
        "graph_k": graph_k,
        "pntk_temperature": pntk_temperature,
        "mu": mu,
        "lam": lam,
        "t_outer": t_outer,
        "t_diffuse": t_diffuse,
        "ta_cd": ta_cd,
        "use_pntk_fast": use_pntk_fast,
        "num_prompts": len(prompt_embeddings),
        "final_beta": beta.tolist(),
        "diffusion_delta_tail": diffusion_deltas[-10:],
        "final_affinity_nnz": int(np.count_nonzero(affinity)),
        "red_execution_device": str(device),
    }


def _require_cuda_device(device, backend_name: str):
    torch = require_module("torch", "pip install torch")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"NTK-SC red_backend='{backend_name}' requires a CUDA device, got '{torch_device}'.")
    return torch_device


def _run_red_cached_fallback(
    *,
    fallback_backend: str,
    image_features: np.ndarray,
    prompt_embeddings: list[np.ndarray],
    cluster_num: int,
    graph_k: int,
    pntk_temperature: float,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    device,
    pntk_block_size: int,
    pntk_column_block_size: int,
    spectral_n_init: int,
    red_block_size: int,
    red_prune_k: int | None,
    max_sparse_nnz: int,
    cache_dir: Path,
    red_graph_cache_key: str,
    force_recompute_red_graphs: bool,
    keep_red_work_dir: bool,
    red_work_dir: Any,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if fallback_backend in {"dense_cached", "dense_cached_gpu_red"}:
        transition_list, transition_cache_paths, transition_cache_hits = _load_or_build_dense_pntk_transition_graphs(
            image_features=image_features,
            prompt_embeddings=prompt_embeddings,
            graph_k=graph_k,
            pntk_temperature=pntk_temperature,
            device=device,
            cache_dir=ensure_dir(cache_dir / "red_dense_graphs"),
            red_graph_cache_key=red_graph_cache_key,
            force_recompute=force_recompute_red_graphs,
            progress=progress,
        )
        if fallback_backend == "dense_cached_gpu_red":
            predictions, metadata = _run_red_out_of_core_gpu(
                transition_list=transition_list,
                cluster_num=cluster_num,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                spectral_n_init=spectral_n_init,
                red_block_size=red_block_size,
                cache_dir=cache_dir,
                red_graph_cache_key=red_graph_cache_key,
                keep_red_work_dir=keep_red_work_dir,
                red_work_dir=red_work_dir,
                device=device,
                progress=progress,
            )
        else:
            predictions, metadata = _run_red_out_of_core(
                transition_list=transition_list,
                cluster_num=cluster_num,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                spectral_n_init=spectral_n_init,
                red_block_size=red_block_size,
                cache_dir=cache_dir,
                red_graph_cache_key=red_graph_cache_key,
                keep_red_work_dir=keep_red_work_dir,
                red_work_dir=red_work_dir,
                progress=progress,
            )
        metadata.update(
            {
                "red_backend": fallback_backend,
                "dense_cached_transition_source": "full_dense_gpu_pntk",
                "pntk_block_size": None,
                "pntk_column_block_size": None,
            }
        )
    else:
        transition_list, transition_cache_paths, transition_cache_hits = _load_or_build_sparse_pntk_transition_graphs(
            image_features=image_features,
            prompt_embeddings=prompt_embeddings,
            graph_k=graph_k,
            pntk_temperature=pntk_temperature,
            pntk_block_size=pntk_block_size,
            pntk_column_block_size=pntk_column_block_size,
            device=device,
            cache_dir=ensure_dir(cache_dir / "red_graphs"),
            red_graph_cache_key=red_graph_cache_key,
            force_recompute=force_recompute_red_graphs,
            progress=progress,
        )
        if fallback_backend == "full_sparse":
            predictions, metadata = _run_red_sparse(
                transition_list=transition_list,
                cluster_num=cluster_num,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                red_prune_k=red_prune_k,
                max_sparse_nnz=max_sparse_nnz,
                spectral_n_init=spectral_n_init,
                progress=progress,
            )
        elif fallback_backend == "out_of_core":
            predictions, metadata = _run_red_out_of_core(
                transition_list=transition_list,
                cluster_num=cluster_num,
                mu=mu,
                lam=lam,
                t_outer=t_outer,
                t_diffuse=t_diffuse,
                ta_cd=ta_cd,
                seed=seed,
                spectral_n_init=spectral_n_init,
                red_block_size=red_block_size,
                cache_dir=cache_dir,
                red_graph_cache_key=red_graph_cache_key,
                keep_red_work_dir=keep_red_work_dir,
                red_work_dir=red_work_dir,
                progress=progress,
            )
        else:
            raise ValueError(f"Unsupported NTK-SC fallback backend '{fallback_backend}'.")
        metadata.update(
            {
                "red_backend": fallback_backend,
                "dense_cached_transition_source": None,
                "pntk_block_size": pntk_block_size,
                "pntk_column_block_size": pntk_column_block_size,
            }
        )

    metadata.update(
        {
            "graph_k": graph_k,
            "pntk_temperature": pntk_temperature,
            "red_block_size": red_block_size,
            "spectral_n_init": spectral_n_init,
            "red_graph_cache_key": red_graph_cache_key,
            "red_transition_cache_paths": [str(path) for path in transition_cache_paths],
            "red_transition_cache_hits": transition_cache_hits,
        }
    )
    return predictions, metadata


def _load_or_build_dense_pntk_transition_graphs(
    *,
    image_features: np.ndarray,
    prompt_embeddings: list[np.ndarray],
    graph_k: int,
    pntk_temperature: float,
    device,
    cache_dir: Path,
    red_graph_cache_key: str,
    force_recompute: bool,
    progress=None,
):
    sparse = require_module("scipy.sparse", "pip install scipy")
    torch = require_module("torch", "pip install torch")
    progress = progress or get_progress_logger("ntk_sc")
    transition_list = []
    cache_paths: list[Path] = []
    cache_hits: list[bool] = []
    images_emb = torch.from_numpy(_normalize_rows(image_features.astype("float32"))).to(device).float()
    kx = images_emb @ images_emb.t()

    for prompt_index, prompt_embedding in enumerate(prompt_embeddings, start=1):
        cache_path = cache_file_path(
            cache_dir,
            f"{red_graph_cache_key}__prompt{prompt_index}__dense_pntk_transition",
            ".npz",
        )
        cache_paths.append(cache_path)
        if cache_path.exists() and not force_recompute:
            progress.log(f"Loading cached dense-pNTK transition graph: {cache_path}")
            transition_list.append(sparse.load_npz(cache_path).tocsr().astype(np.float32))
            cache_hits.append(True)
            progress.step("Building dense-pNTK cached transition graphs", prompt_index, len(prompt_embeddings), noun="prompt")
            continue

        nouns_emb = torch.from_numpy(_normalize_rows(prompt_embedding.astype("float32"))).to(device).float()
        with torch.no_grad():
            kernel = _compute_pntk_matrix_fast(images_emb, nouns_emb, pntk_temperature, kx_precomputed=kx)
        transition = _build_sparse_knn_graph_from_similarity(kernel, k=graph_k)
        sparse.save_npz(cache_path, transition)
        progress.log(f"Saved dense-pNTK transition graph: {cache_path}")
        transition_list.append(transition)
        cache_hits.append(False)
        del nouns_emb, kernel, transition
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.step("Building dense-pNTK cached transition graphs", prompt_index, len(prompt_embeddings), noun="prompt")

    del images_emb, kx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transition_list, cache_paths, cache_hits


def _build_sparse_knn_graph_from_similarity(similarity: np.ndarray, k: int):
    sparse = require_module("scipy.sparse", "pip install scipy")
    node_count = similarity.shape[0]
    if node_count <= 1:
        return sparse.csr_matrix((node_count, node_count), dtype=np.float32)

    np.fill_diagonal(similarity, 0.0)
    effective_k = min(k, node_count - 1)
    indices = np.argpartition(-similarity, kth=effective_k - 1, axis=1)[:, :effective_k]
    rows = np.repeat(np.arange(node_count, dtype=np.int64), indices.shape[1])
    cols = indices.reshape(-1).astype(np.int64, copy=False)
    values = similarity[rows, cols].astype(np.float32, copy=False)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    positive = values > 0.0
    adjacency = sparse.csr_matrix(
        (values[positive], (rows[positive], cols[positive])),
        shape=(node_count, node_count),
        dtype=np.float32,
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.eliminate_zeros()
    return _transition_from_sparse_w(adjacency)


def _load_or_build_sparse_pntk_transition_graphs(
    *,
    image_features: np.ndarray,
    prompt_embeddings: list[np.ndarray],
    graph_k: int,
    pntk_temperature: float,
    pntk_block_size: int,
    pntk_column_block_size: int,
    device,
    cache_dir: Path,
    red_graph_cache_key: str,
    force_recompute: bool,
    progress=None,
):
    sparse = require_module("scipy.sparse", "pip install scipy")
    progress = progress or get_progress_logger("ntk_sc")
    transition_list = []
    cache_paths: list[Path] = []
    cache_hits: list[bool] = []
    for prompt_index, prompt_embedding in enumerate(prompt_embeddings, start=1):
        cache_path = cache_file_path(
            cache_dir,
            f"{red_graph_cache_key}__prompt{prompt_index}__transition",
            ".npz",
        )
        cache_paths.append(cache_path)
        if cache_path.exists() and not force_recompute:
            progress.log(f"Loading cached sparse pNTK transition graph: {cache_path}")
            transition_list.append(sparse.load_npz(cache_path).tocsr().astype(np.float32))
            cache_hits.append(True)
            progress.step("Building sparse pNTK transition graphs", prompt_index, len(prompt_embeddings), noun="prompt")
            continue

        transition = _build_sparse_pntk_transition_graph(
            image_features=image_features,
            prompt_embedding=prompt_embedding,
            graph_k=graph_k,
            pntk_temperature=pntk_temperature,
            pntk_block_size=pntk_block_size,
            pntk_column_block_size=pntk_column_block_size,
            device=device,
            progress=progress,
            prompt_index=prompt_index,
            prompt_count=len(prompt_embeddings),
        )
        sparse.save_npz(cache_path, transition)
        progress.log(f"Saved sparse pNTK transition graph: {cache_path}")
        transition_list.append(transition)
        cache_hits.append(False)
        progress.step("Building sparse pNTK transition graphs", prompt_index, len(prompt_embeddings), noun="prompt")
    return transition_list, cache_paths, cache_hits


def _build_sparse_pntk_transition_graph(
    *,
    image_features: np.ndarray,
    prompt_embedding: np.ndarray,
    graph_k: int,
    pntk_temperature: float,
    pntk_block_size: int,
    pntk_column_block_size: int,
    device,
    progress,
    prompt_index: int,
    prompt_count: int,
):
    torch = require_module("torch", "pip install torch")
    sparse = require_module("scipy.sparse", "pip install scipy")

    normalized_images = _normalize_rows(image_features.astype("float32"))
    normalized_nouns = _normalize_rows(prompt_embedding.astype("float32"))
    sample_count = normalized_images.shape[0]
    if sample_count <= 1:
        return sparse.csr_matrix((sample_count, sample_count), dtype=np.float32)

    effective_k = min(graph_k, sample_count - 1)
    images_emb = torch.from_numpy(normalized_images).to(device).float()
    nouns_emb = torch.from_numpy(normalized_nouns).to(device).float()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    total_row_blocks = max(1, (sample_count + pntk_block_size - 1) // pntk_block_size)

    with torch.no_grad():
        for row_block_index, row_start in enumerate(range(0, sample_count, pntk_block_size), start=1):
            row_end = min(row_start + pntk_block_size, sample_count)
            image_rows = images_emb[row_start:row_end]
            row_count = row_end - row_start
            row_top_values = torch.full((row_count, effective_k), -float("inf"), device=device)
            row_top_indices = torch.full((row_count, effective_k), -1, dtype=torch.long, device=device)
            soft_rows = torch.softmax((image_rows @ nouns_emb.t()) / pntk_temperature, dim=1)

            for col_start in range(0, sample_count, pntk_column_block_size):
                col_end = min(col_start + pntk_column_block_size, sample_count)
                image_cols = images_emb[col_start:col_end]
                soft_cols = torch.softmax((image_cols @ nouns_emb.t()) / pntk_temperature, dim=1)
                kx_block = image_rows @ image_cols.t()
                ks_block = soft_rows @ soft_cols.t()
                kernel_block = (kx_block * ks_block) / (pntk_temperature * pntk_temperature)

                diag_start = max(row_start, col_start)
                diag_end = min(row_end, col_end)
                if diag_start < diag_end:
                    local_rows = torch.arange(diag_start - row_start, diag_end - row_start, device=device)
                    local_cols = torch.arange(diag_start - col_start, diag_end - col_start, device=device)
                    kernel_block[local_rows, local_cols] = -float("inf")

                chunk_k = min(effective_k, col_end - col_start)
                chunk_values, chunk_indices = torch.topk(kernel_block, k=chunk_k, dim=1)
                chunk_indices = chunk_indices + col_start
                merged_values = torch.cat((row_top_values, chunk_values), dim=1)
                merged_indices = torch.cat((row_top_indices, chunk_indices), dim=1)
                row_top_values, selection = torch.topk(merged_values, k=effective_k, dim=1)
                row_top_indices = torch.gather(merged_indices, dim=1, index=selection)
                del image_cols, soft_cols, kx_block, ks_block, kernel_block, chunk_values, chunk_indices

            row_values = row_top_values.detach().cpu().numpy().astype(np.float32, copy=False)
            row_indices = row_top_indices.detach().cpu().numpy().astype(np.int64, copy=False)
            np.nan_to_num(row_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            row_values = np.maximum(row_values, 0.0)
            valid_mask = row_indices >= 0
            block_rows = np.repeat(np.arange(row_start, row_end, dtype=np.int64), effective_k)[valid_mask.reshape(-1)]
            rows.append(block_rows)
            cols.append(row_indices.reshape(-1)[valid_mask.reshape(-1)])
            vals.append(row_values.reshape(-1)[valid_mask.reshape(-1)])
            progress.step(
                f"Prompt {prompt_index}/{prompt_count} pNTK row blocks",
                row_block_index,
                total_row_blocks,
                noun="block",
            )
            del image_rows, soft_rows, row_top_values, row_top_indices

    adjacency = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(sample_count, sample_count),
        dtype=np.float32,
    )
    adjacency.eliminate_zeros()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.eliminate_zeros()
    del images_emb, nouns_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _transition_from_sparse_w(adjacency)


def _transition_from_sparse_w(adjacency):
    adjacency = adjacency.tocsr().astype(np.float32, copy=False)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    scale = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    transition = adjacency.multiply(scale[:, None]).multiply(scale[None, :]).tocsr()
    transition.eliminate_zeros()
    return transition.astype(np.float32, copy=False)


def _run_red_sparse(
    *,
    transition_list,
    cluster_num: int,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    red_prune_k: int | None,
    max_sparse_nnz: int,
    spectral_n_init: int,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    sparse = require_module("scipy.sparse", "pip install scipy")

    progress = progress or get_progress_logger("ntk_sc")
    sample_count = transition_list[0].shape[0]
    identity = sparse.identity(sample_count, format="csr", dtype=np.float32)
    adjacency = identity.copy()
    transition_transposes = [transition.T.tocsr() for transition in transition_list]
    beta = np.ones(len(transition_list), dtype=np.float32) / len(transition_list)
    diffusion_deltas: list[float] = []

    for outer_index in range(t_outer):
        alpha = beta / (mu + float(beta.sum()))
        alpha_sum = float(alpha.sum())
        previous = adjacency
        for _ in range(t_diffuse):
            aggregated = sparse.csr_matrix((sample_count, sample_count), dtype=np.float32)
            for index, transition in enumerate(transition_list):
                diffusion = transition @ adjacency @ transition_transposes[index]
                aggregated = aggregated + diffusion.multiply(float(alpha[index]))
            adjacency = aggregated + identity.multiply(1.0 - alpha_sum)
            adjacency = (adjacency + adjacency.T).multiply(0.5).tocsr()
            _clip_sparse_nonnegative(adjacency)
            if red_prune_k is not None:
                adjacency = _prune_sparse_rows(adjacency, red_prune_k)
                adjacency = adjacency.maximum(adjacency.T).tocsr()
                adjacency.eliminate_zeros()
            _check_sparse_nnz(adjacency, max_sparse_nnz)
            delta = _sparse_relative_delta(adjacency, previous)
            diffusion_deltas.append(delta)
            if delta < 1e-4:
                break
            previous = adjacency

        adjacency_norm_sq = _sparse_frobenius_norm_squared(adjacency)
        hv_values = np.asarray(
            [
                _compute_hv_sparse(adjacency, transition, transition_transposes[index], adjacency_norm_sq)
                for index, transition in enumerate(transition_list)
            ],
            dtype=np.float32,
        )
        beta = _update_beta_coordinate_descent(beta, hv_values, lam, passes=ta_cd)
        progress.step("Sparse RED diffusion", outer_index + 1, t_outer, noun="outer step")

    predictions = _run_sparse_spectral_clustering(
        adjacency,
        n_clusters=cluster_num,
        random_state=seed,
        n_init=spectral_n_init,
    )
    return predictions.astype(np.int64), {
        "graph_k": None,
        "pntk_temperature": None,
        "mu": mu,
        "lam": lam,
        "t_outer": t_outer,
        "t_diffuse": t_diffuse,
        "ta_cd": ta_cd,
        "use_pntk_fast": True,
        "num_prompts": len(transition_list),
        "final_beta": beta.tolist(),
        "diffusion_delta_tail": diffusion_deltas[-10:],
        "final_affinity_nnz": int(adjacency.nnz),
        "final_affinity_density": float(adjacency.nnz / (sample_count * sample_count)),
    }


def _run_sparse_spectral_clustering(affinity_matrix, *, n_clusters: int, random_state: int, n_init: int) -> np.ndarray:
    sparse = require_module("scipy.sparse", "pip install scipy")
    sparse_linalg = require_module("scipy.sparse.linalg", "pip install scipy")
    sklearn_utils = require_module("sklearn.utils", "pip install scikit-learn")

    affinity_matrix = sklearn_utils.check_symmetric(affinity_matrix)
    laplacian = sparse.csgraph.laplacian(affinity_matrix, normed=True)
    _, eigenvectors = sparse_linalg.eigsh(
        sparse.identity(laplacian.shape[0], dtype=np.float32) - laplacian,
        k=n_clusters,
        which="LA",
    )
    return _assign_spectral_labels(eigenvectors, n_clusters=n_clusters, random_state=random_state, n_init=n_init)


def _assign_spectral_labels(
    eigenvectors: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> np.ndarray:
    sklearn_preprocessing = require_module("sklearn.preprocessing", "pip install scikit-learn")
    sklearn_utils = require_module("sklearn.utils", "pip install scikit-learn")
    random_state_obj = sklearn_utils.check_random_state(random_state)
    try:
        spectral_module = require_module("sklearn.cluster._spectral", "pip install scikit-learn")
        labels = spectral_module.discretize(eigenvectors, random_state=random_state_obj)
        return labels.astype(np.int64, copy=False)
    except Exception:
        embedding = sklearn_preprocessing.normalize(eigenvectors)
        labels, _ = run_faiss_kmeans(
            embedding,
            n_clusters=n_clusters,
            n_iter=300,
            n_redo=n_init,
            random_state=random_state,
            spherical=False,
        )
        return labels.astype(np.int64, copy=False)


def _clip_sparse_nonnegative(matrix) -> None:
    if matrix.nnz == 0:
        return
    matrix.data[~np.isfinite(matrix.data)] = 0.0
    matrix.data[matrix.data < 0.0] = 0.0
    matrix.eliminate_zeros()


def _check_sparse_nnz(matrix, max_sparse_nnz: int) -> None:
    if int(matrix.nnz) <= max_sparse_nnz:
        return
    raise RuntimeError(
        "NTK-SC full_sparse RED expanded beyond max_sparse_nnz. "
        f"nnz={matrix.nnz}, max_sparse_nnz={max_sparse_nnz}. "
        "Use method.red_backend=out_of_core for the exact dense RED path, or set method.red_prune_k for an approximate sparse run."
    )


def _sparse_frobenius_norm_squared(matrix) -> float:
    return float(matrix.multiply(matrix).sum())


def _sparse_relative_delta(current, previous) -> float:
    numerator = np.sqrt(max(_sparse_frobenius_norm_squared(current - previous), 0.0))
    denominator = np.sqrt(max(_sparse_frobenius_norm_squared(previous), 0.0)) + 1e-12
    return float(numerator / denominator)


def _compute_hv_sparse(adjacency, transition, transition_t, adjacency_norm_sq: float) -> float:
    diffusion = transition @ adjacency @ transition_t
    return float(adjacency_norm_sq - adjacency.multiply(diffusion).sum())


def _prune_sparse_rows(matrix, keep_k: int):
    sparse = require_module("scipy.sparse", "pip install scipy")
    matrix = matrix.tocsr()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for row_index in range(matrix.shape[0]):
        start = matrix.indptr[row_index]
        end = matrix.indptr[row_index + 1]
        if start == end:
            continue
        row_data = matrix.data[start:end]
        row_cols = matrix.indices[start:end]
        if row_data.shape[0] > keep_k:
            selected = np.argpartition(-row_data, keep_k - 1)[:keep_k]
            row_data = row_data[selected]
            row_cols = row_cols[selected]
        rows.append(np.full(row_data.shape[0], row_index, dtype=np.int64))
        cols.append(row_cols.astype(np.int64, copy=False))
        vals.append(row_data.astype(np.float32, copy=False))
    if not vals:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    pruned = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape,
        dtype=np.float32,
    )
    pruned.eliminate_zeros()
    return pruned


def _run_red_out_of_core(
    *,
    transition_list,
    cluster_num: int,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    spectral_n_init: int,
    red_block_size: int,
    cache_dir: Path,
    red_graph_cache_key: str,
    keep_red_work_dir: bool,
    red_work_dir: Any,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    progress = progress or get_progress_logger("ntk_sc")
    sample_count = transition_list[0].shape[0]
    temp_context = None
    if red_work_dir is None:
        if keep_red_work_dir:
            work_path = ensure_dir(cache_dir / "red_out_of_core" / red_graph_cache_key)
        else:
            temp_context = tempfile.TemporaryDirectory(
                prefix="ntk_sc_red_",
                dir=ensure_dir(cache_dir / "red_out_of_core_tmp"),
            )
            work_path = Path(temp_context.name)
    else:
        work_path = ensure_dir(Path(red_work_dir).expanduser())

    try:
        current = _create_red_memmap(work_path / "adjacency_a.dat", sample_count, red_block_size, identity=True)
        target = _create_red_memmap(work_path / "adjacency_b.dat", sample_count, red_block_size, identity=False)
        beta = np.ones(len(transition_list), dtype=np.float32) / len(transition_list)
        diffusion_deltas: list[float] = []

        for outer_index in range(t_outer):
            alpha = beta / (mu + float(beta.sum()))
            alpha_sum = float(alpha.sum())
            for _ in range(t_diffuse):
                _write_out_of_core_red_update(
                    previous=current,
                    target=target,
                    transition_list=transition_list,
                    alpha=alpha,
                    alpha_sum=alpha_sum,
                    block_size=red_block_size,
                )
                _symmetrize_memmap_in_place(target, red_block_size)
                delta = _memmap_relative_delta(target, current, red_block_size)
                diffusion_deltas.append(delta)
                current, target = target, current
                if delta < 1e-4:
                    break

            adjacency_norm_sq = _memmap_frobenius_norm_squared(current, red_block_size)
            hv_values = np.asarray(
                [
                    _compute_hv_out_of_core(current, transition, adjacency_norm_sq, red_block_size)
                    for transition in transition_list
                ],
                dtype=np.float32,
            )
            beta = _update_beta_coordinate_descent(beta, hv_values, lam, passes=ta_cd)
            progress.step("Out-of-core RED diffusion", outer_index + 1, t_outer, noun="outer step")

        predictions = _run_out_of_core_spectral_clustering(
            current,
            n_clusters=cluster_num,
            random_state=seed,
            n_init=spectral_n_init,
            block_size=red_block_size,
        )
        metadata = {
            "graph_k": None,
            "pntk_temperature": None,
            "mu": mu,
            "lam": lam,
            "t_outer": t_outer,
            "t_diffuse": t_diffuse,
            "ta_cd": ta_cd,
            "use_pntk_fast": True,
            "num_prompts": len(transition_list),
            "final_beta": beta.tolist(),
            "diffusion_delta_tail": diffusion_deltas[-10:],
            "red_work_dir": str(work_path) if keep_red_work_dir or red_work_dir is not None else None,
            "final_affinity_nnz": int(sample_count * sample_count),
            "final_affinity_density": 1.0,
        }
        return predictions.astype(np.int64), metadata
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def _run_red_out_of_core_gpu(
    *,
    transition_list,
    cluster_num: int,
    mu: float,
    lam: float,
    t_outer: int,
    t_diffuse: int,
    ta_cd: int,
    seed: int,
    spectral_n_init: int,
    red_block_size: int,
    cache_dir: Path,
    red_graph_cache_key: str,
    keep_red_work_dir: bool,
    red_work_dir: Any,
    device,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = require_module("torch", "pip install torch")
    progress = progress or get_progress_logger("ntk_sc")
    device = _require_cuda_device(device, "dense_cached_gpu_red")
    sample_count = transition_list[0].shape[0]
    transition_gpu_list = [_torch_sparse_from_scipy(transition, device) for transition in transition_list]
    temp_context = None
    if red_work_dir is None:
        if keep_red_work_dir:
            work_path = ensure_dir(cache_dir / "red_out_of_core_gpu" / red_graph_cache_key)
        else:
            temp_context = tempfile.TemporaryDirectory(
                prefix="ntk_sc_red_gpu_",
                dir=ensure_dir(cache_dir / "red_out_of_core_gpu_tmp"),
            )
            work_path = Path(temp_context.name)
    else:
        work_path = ensure_dir(Path(red_work_dir).expanduser())

    try:
        current = _create_red_memmap(work_path / "adjacency_a.dat", sample_count, red_block_size, identity=True)
        target = _create_red_memmap(work_path / "adjacency_b.dat", sample_count, red_block_size, identity=False)
        beta = np.ones(len(transition_list), dtype=np.float32) / len(transition_list)
        diffusion_deltas: list[float] = []

        with torch.no_grad():
            for outer_index in range(t_outer):
                alpha = beta / (mu + float(beta.sum()))
                alpha_sum = float(alpha.sum())
                for _ in range(t_diffuse):
                    _write_out_of_core_red_update_gpu(
                        previous=current,
                        target=target,
                        transition_list=transition_list,
                        transition_gpu_list=transition_gpu_list,
                        alpha=alpha,
                        alpha_sum=alpha_sum,
                        block_size=red_block_size,
                        device=device,
                    )
                    _symmetrize_memmap_in_place(target, red_block_size)
                    delta = _memmap_relative_delta(target, current, red_block_size)
                    diffusion_deltas.append(delta)
                    current, target = target, current
                    if delta < 1e-4:
                        break

                adjacency_norm_sq = _memmap_frobenius_norm_squared(current, red_block_size)
                hv_values = np.asarray(
                    [
                        _compute_hv_out_of_core_gpu(
                            current,
                            transition,
                            transition_gpu_list[index],
                            adjacency_norm_sq,
                            red_block_size,
                            device,
                        )
                        for index, transition in enumerate(transition_list)
                    ],
                    dtype=np.float32,
                )
                beta = _update_beta_coordinate_descent(beta, hv_values, lam, passes=ta_cd)
                progress.step("GPU out-of-core RED diffusion", outer_index + 1, t_outer, noun="outer step")

        predictions = _run_out_of_core_spectral_clustering(
            current,
            n_clusters=cluster_num,
            random_state=seed,
            n_init=spectral_n_init,
            block_size=red_block_size,
        )
        metadata = {
            "graph_k": None,
            "pntk_temperature": None,
            "mu": mu,
            "lam": lam,
            "t_outer": t_outer,
            "t_diffuse": t_diffuse,
            "ta_cd": ta_cd,
            "use_pntk_fast": True,
            "num_prompts": len(transition_list),
            "final_beta": beta.tolist(),
            "diffusion_delta_tail": diffusion_deltas[-10:],
            "red_work_dir": str(work_path) if keep_red_work_dir or red_work_dir is not None else None,
            "final_affinity_nnz": int(sample_count * sample_count),
            "final_affinity_density": 1.0,
            "red_execution_device": str(device),
        }
        return predictions.astype(np.int64), metadata
    finally:
        if temp_context is not None:
            temp_context.cleanup()
        del transition_gpu_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _create_red_memmap(path: Path, sample_count: int, block_size: int, *, identity: bool):
    matrix = np.memmap(path, dtype="float32", mode="w+", shape=(sample_count, sample_count))
    for start in range(0, sample_count, block_size):
        stop = min(start + block_size, sample_count)
        matrix[start:stop, :] = 0.0
    if identity:
        indices = np.arange(sample_count)
        matrix[indices, indices] = 1.0
    matrix.flush()
    return matrix


def _write_out_of_core_red_update(*, previous, target, transition_list, alpha: np.ndarray, alpha_sum: float, block_size: int):
    sample_count = previous.shape[0]
    for col_start in range(0, sample_count, block_size):
        col_stop = min(col_start + block_size, sample_count)
        block = np.zeros((sample_count, col_stop - col_start), dtype=np.float32)
        for index, transition in enumerate(transition_list):
            block += float(alpha[index]) * _sparse_quadratic_memmap_columns(
                transition,
                previous,
                col_start,
                col_stop,
            )
        for local_col, global_col in enumerate(range(col_start, col_stop)):
            block[global_col, local_col] += 1.0 - alpha_sum
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(block, 0.0, out=block)
        target[:, col_start:col_stop] = block
    target.flush()


def _write_out_of_core_red_update_gpu(
    *,
    previous,
    target,
    transition_list,
    transition_gpu_list,
    alpha: np.ndarray,
    alpha_sum: float,
    block_size: int,
    device,
) -> None:
    torch = require_module("torch", "pip install torch")
    sample_count = previous.shape[0]
    for col_start in range(0, sample_count, block_size):
        col_stop = min(col_start + block_size, sample_count)
        width = col_stop - col_start
        block_gpu = torch.zeros((sample_count, width), dtype=torch.float32, device=device)
        for index, transition in enumerate(transition_list):
            diffusion_gpu = _sparse_quadratic_memmap_columns_gpu(
                transition,
                transition_gpu_list[index],
                previous,
                col_start,
                col_stop,
                device,
            )
            block_gpu.add_(diffusion_gpu, alpha=float(alpha[index]))
            del diffusion_gpu
        diag_rows = torch.arange(col_start, col_stop, dtype=torch.long, device=device)
        diag_cols = torch.arange(width, dtype=torch.long, device=device)
        block_gpu[diag_rows, diag_cols] += 1.0 - alpha_sum
        block_gpu = torch.nan_to_num(block_gpu, nan=0.0, posinf=0.0, neginf=0.0)
        block_gpu.clamp_(min=0.0)
        target[:, col_start:col_stop] = block_gpu.detach().cpu().numpy()
        del block_gpu
    target.flush()


def _symmetrize_memmap_in_place(matrix, block_size: int) -> None:
    sample_count = matrix.shape[0]
    for row_start in range(0, sample_count, block_size):
        row_stop = min(row_start + block_size, sample_count)
        for col_start in range(row_start, sample_count, block_size):
            col_stop = min(col_start + block_size, sample_count)
            block = np.asarray(matrix[row_start:row_stop, col_start:col_stop], dtype=np.float32).copy()
            if row_start == col_start:
                sym_block = 0.5 * (block + block.T)
                np.maximum(sym_block, 0.0, out=sym_block)
                matrix[row_start:row_stop, col_start:col_stop] = sym_block
            else:
                transpose_block = np.asarray(matrix[col_start:col_stop, row_start:row_stop], dtype=np.float32).copy()
                sym_block = 0.5 * (block + transpose_block.T)
                np.maximum(sym_block, 0.0, out=sym_block)
                matrix[row_start:row_stop, col_start:col_stop] = sym_block
                matrix[col_start:col_stop, row_start:row_stop] = sym_block.T
    matrix.flush()


def _sparse_quadratic_memmap_columns(transition, matrix, col_start: int, col_stop: int) -> np.ndarray:
    right_product = _memmap_times_transition_transpose_columns(matrix, transition, col_start, col_stop)
    return np.asarray(transition @ right_product, dtype=np.float32)


def _sparse_quadratic_memmap_columns_gpu(transition, transition_gpu, matrix, col_start: int, col_stop: int, device):
    torch = require_module("torch", "pip install torch")
    right_product = _memmap_times_transition_transpose_columns_gpu(matrix, transition, col_start, col_stop, device)
    return torch.sparse.mm(transition_gpu, right_product)


def _memmap_times_transition_transpose_columns(matrix, transition, col_start: int, col_stop: int) -> np.ndarray:
    sample_count = matrix.shape[0]
    output = np.zeros((sample_count, col_stop - col_start), dtype=np.float32)
    for local_col, transition_row in enumerate(range(col_start, col_stop)):
        start = transition.indptr[transition_row]
        stop = transition.indptr[transition_row + 1]
        if start == stop:
            continue
        indices = transition.indices[start:stop]
        values = transition.data[start:stop].astype(np.float32, copy=False)
        output[:, local_col] = np.asarray(matrix[:, indices]) @ values
    return output


def _memmap_times_transition_transpose_columns_gpu(matrix, transition, col_start: int, col_stop: int, device):
    torch = require_module("torch", "pip install torch")
    sample_count = matrix.shape[0]
    output = torch.empty((sample_count, col_stop - col_start), dtype=torch.float32, device=device)
    for local_col, transition_row in enumerate(range(col_start, col_stop)):
        start = transition.indptr[transition_row]
        stop = transition.indptr[transition_row + 1]
        if start == stop:
            output[:, local_col].zero_()
            continue
        indices = transition.indices[start:stop]
        values = transition.data[start:stop].astype(np.float32, copy=False)
        columns = np.ascontiguousarray(np.asarray(matrix[:, indices], dtype=np.float32))
        columns_gpu = torch.from_numpy(columns).to(device)
        values_gpu = torch.from_numpy(values).to(device)
        output[:, local_col] = columns_gpu @ values_gpu
        del columns_gpu, values_gpu
    return output


def _memmap_frobenius_norm_squared(matrix, block_size: int) -> float:
    total = 0.0
    for start in range(0, matrix.shape[0], block_size):
        stop = min(start + block_size, matrix.shape[0])
        block = np.asarray(matrix[start:stop, :], dtype=np.float32)
        total += float(np.sum(block * block))
    return total


def _memmap_relative_delta(current, previous, block_size: int) -> float:
    numerator = 0.0
    denominator = 0.0
    for start in range(0, current.shape[0], block_size):
        stop = min(start + block_size, current.shape[0])
        current_block = np.asarray(current[start:stop, :], dtype=np.float32)
        previous_block = np.asarray(previous[start:stop, :], dtype=np.float32)
        diff = current_block - previous_block
        numerator += float(np.sum(diff * diff))
        denominator += float(np.sum(previous_block * previous_block))
    return float(np.sqrt(numerator) / (np.sqrt(denominator) + 1e-12))


def _compute_hv_out_of_core(matrix, transition, adjacency_norm_sq: float, block_size: int) -> float:
    cross = 0.0
    for col_start in range(0, matrix.shape[0], block_size):
        col_stop = min(col_start + block_size, matrix.shape[0])
        diffusion_block = _sparse_quadratic_memmap_columns(transition, matrix, col_start, col_stop)
        adjacency_block = np.asarray(matrix[:, col_start:col_stop], dtype=np.float32)
        cross += float(np.sum(adjacency_block * diffusion_block))
    return float(adjacency_norm_sq - cross)


def _compute_hv_out_of_core_gpu(matrix, transition, transition_gpu, adjacency_norm_sq: float, block_size: int, device) -> float:
    torch = require_module("torch", "pip install torch")
    cross = 0.0
    for col_start in range(0, matrix.shape[0], block_size):
        col_stop = min(col_start + block_size, matrix.shape[0])
        diffusion_gpu = _sparse_quadratic_memmap_columns_gpu(
            transition,
            transition_gpu,
            matrix,
            col_start,
            col_stop,
            device,
        )
        adjacency_block = np.ascontiguousarray(np.asarray(matrix[:, col_start:col_stop], dtype=np.float32))
        adjacency_block_gpu = torch.from_numpy(adjacency_block).to(device)
        cross += float(torch.sum(adjacency_block_gpu * diffusion_gpu).detach().cpu())
        del diffusion_gpu, adjacency_block_gpu
    return float(adjacency_norm_sq - cross)


def _torch_sparse_from_scipy(matrix, device):
    torch = require_module("torch", "pip install torch")
    coo = matrix.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    sparse_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device),
        torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(device),
        size=coo.shape,
        dtype=torch.float32,
        device=device,
    )
    return sparse_tensor.coalesce()


def _run_out_of_core_spectral_clustering(
    affinity_memmap,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int,
    block_size: int,
) -> np.ndarray:
    sparse_linalg = require_module("scipy.sparse.linalg", "pip install scipy")

    sample_count = affinity_memmap.shape[0]
    degree = np.zeros(sample_count, dtype=np.float64)
    for start in range(0, sample_count, block_size):
        stop = min(start + block_size, sample_count)
        degree[start:stop] = np.asarray(affinity_memmap[start:stop, :], dtype=np.float64).sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.clip(degree, a_min=1e-12, a_max=None))

    def matvec(vector):
        weighted = inv_sqrt_degree * vector
        output = np.zeros(sample_count, dtype=np.float64)
        for start in range(0, sample_count, block_size):
            stop = min(start + block_size, sample_count)
            output[start:stop] = np.asarray(affinity_memmap[start:stop, :], dtype=np.float64) @ weighted
        return inv_sqrt_degree * output

    operator = sparse_linalg.LinearOperator(
        shape=(sample_count, sample_count),
        matvec=matvec,
        dtype=np.float64,
    )
    _, eigenvectors = sparse_linalg.eigsh(operator, k=n_clusters, which="LA")
    return _assign_spectral_labels(eigenvectors, n_clusters=n_clusters, random_state=random_state, n_init=n_init)
