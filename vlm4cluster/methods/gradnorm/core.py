from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.feature_cache import (
    cache_file_path,
    cache_key,
    image_dataset_cache_key,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
    wordnet_source_token,
)
from vlm4cluster.methods.tac.core import (
    _encode_image_dataset,
    _encode_wordnet_nouns,
    _normalize_dataset_name,
    _resolve_domain_shift_datasets,
    _resolve_tac_wordnet_csv,
    _retrieve_text_counterparts,
    _run_kmeans,
    _normalize_rows,
)
from vlm4cluster.models import load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import FAISS_KMEANS_CACHE_TOKEN
from vlm4cluster.utils.progress import get_progress_logger


DEFAULT_GRADNORM_FILTER_TEMPERATURE = 0.04
DEFAULT_GRADNORM_RETRIEVAL_TEMPERATURE = 0.005

GRADNORM_FILTER_CLUSTER_DEFAULTS: dict[str, int] = {
    "cifar10": 250,
    "cifar20": 200,
    "stl10": 30,
    "imagenet10": 50,
    "imagenetdogs": 50,
    "dtd": 180,
    "ucf101": 303,
    "imagenet": 2000,
}

GRADNORM_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

GRADNORM_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10"): {
        "filter_cluster_num": 200,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitb32", "cifar20"): {"filter_temperature": 0.005, "retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "cifar100"): {"filter_temperature": 0.005, "retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "stl10"): {"filter_temperature": 0.02, "retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenet10"): {
        "filter_cluster_num": 45,
        "filter_temperature": 0.03,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitb32", "imagenetdogs"): {
        "filter_cluster_num": 60,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb32", "dtd"): {
        "filter_cluster_num": 141,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb32", "ucf101"): {"filter_temperature": 0.01, "retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "imagenet"): {"filter_temperature": 0.02, "retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "places365standard"): {
        "filter_cluster_num": 1500,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitb32", "aircraft"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb32", "cars"): {"filter_cluster_num": 250, "retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "flowers"): {
        "filter_cluster_num": 200,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb32", "food"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb32", "pets"): {
        "filter_cluster_num": 100,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.03,
    },
    ("laion400m", "vitb16", "cifar10"): {
        "filter_cluster_num": 200,
        "filter_temperature": 0.05,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitb16", "cifar20"): {
        "filter_cluster_num": 300,
        "filter_temperature": 0.005,
        "retrieval_temperature": 0.001,
    },
    ("laion400m", "vitb16", "cifar100"): {"filter_temperature": 0.003, "retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "stl10"): {"filter_temperature": 0.02, "retrieval_temperature": 0.03},
    ("laion400m", "vitb16", "imagenet10"): {
        "filter_cluster_num": 45,
        "filter_temperature": 0.02,
        "retrieval_temperature": 0.05,
    },
    ("laion400m", "vitb16", "imagenetdogs"): {
        "filter_cluster_num": 60,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb16", "dtd"): {"filter_temperature": 0.03, "retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "ucf101"): {"filter_cluster_num": 350, "retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "imagenet"): {"filter_temperature": 0.02, "retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "places365standard"): {
        "filter_cluster_num": 1500,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb16", "aircraft"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.02,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitb16", "cars"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.03,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitb16", "flowers"): {
        "filter_cluster_num": 200,
        "filter_temperature": 0.03,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitb16", "food"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.02,
    },
    ("laion400m", "vitb16", "pets"): {
        "filter_cluster_num": 100,
        "filter_temperature": 0.03,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitl14", "cifar10"): {
        "filter_cluster_num": 200,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitl14", "cifar20"): {
        "filter_cluster_num": 300,
        "filter_temperature": 0.005,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitl14", "cifar100"): {"filter_temperature": 0.01, "retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "stl10"): {"filter_temperature": 0.005, "retrieval_temperature": 0.04},
    ("laion400m", "vitl14", "imagenet10"): {
        "filter_cluster_num": 45,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.01,
    },
    ("laion400m", "vitl14", "imagenetdogs"): {
        "filter_cluster_num": 60,
        "filter_temperature": 0.05,
        "retrieval_temperature": 0.03,
    },
    ("laion400m", "vitl14", "dtd"): {"filter_temperature": 0.01, "retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "ucf101"): {
        "filter_cluster_num": 350,
        "filter_temperature": 0.06,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitl14", "imagenet"): {"filter_temperature": 0.01, "retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "places365standard"): {
        "filter_cluster_num": 1500,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.03,
    },
    ("laion400m", "vitl14", "aircraft"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.02,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitl14", "cars"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.01,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitl14", "flowers"): {"filter_cluster_num": 200, "retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "food"): {
        "filter_cluster_num": 250,
        "filter_temperature": 0.02,
        "retrieval_temperature": 0.04,
    },
    ("laion400m", "vitl14", "pets"): {
        "filter_cluster_num": 100,
        "filter_temperature": 0.03,
        "retrieval_temperature": 0.06,
    },
}


@dataclass(slots=True)
class GradNormOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


def _normalize_gradnorm_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_gradnorm_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in GRADNORM_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _resolve_gradnorm_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_gradnorm_profile_token(openclip_pretraining),
        _normalize_gradnorm_profile_token(openclip_backbone),
        _normalize_gradnorm_profile_dataset(dataset_name),
    )
    return dict(GRADNORM_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_gradnorm_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    profile_key = (
        _normalize_gradnorm_profile_token(openclip_pretraining),
        _normalize_gradnorm_profile_token(openclip_backbone),
        _normalize_gradnorm_profile_dataset(dataset_name),
    )
    if profile_key not in GRADNORM_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _default_gradnorm_filter_cluster_num(dataset_name: str, cluster_num: int) -> tuple[int, str]:
    normalized = _normalize_gradnorm_profile_dataset(dataset_name)
    if normalized in GRADNORM_FILTER_CLUSTER_DEFAULTS:
        return GRADNORM_FILTER_CLUSTER_DEFAULTS[normalized], "dataset_default"
    return int(max(cluster_num * 3, cluster_num)), "inferred_default_3x_clusters"


def _resolve_gradnorm_filter_cluster_num(
    params: dict[str, Any],
    defaults: dict[str, Any],
    dataset_name: str,
    cluster_num: int,
) -> tuple[int, str]:
    if "filter_cluster_num" in params:
        return int(params["filter_cluster_num"]), "explicit"
    if "proxy_cluster_num" in params:
        return int(params["proxy_cluster_num"]), "explicit"
    if "filter_cluster_num" in defaults:
        return int(defaults["filter_cluster_num"]), "profile"
    return _default_gradnorm_filter_cluster_num(dataset_name, cluster_num)


def _resolve_gradnorm_filter_temperature(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "filter_temperature" in params:
        return float(params["filter_temperature"])
    if "temp" in params:
        return float(params["temp"])
    return float(defaults.get("filter_temperature", DEFAULT_GRADNORM_FILTER_TEMPERATURE))


def _resolve_gradnorm_retrieval_temperature(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "retrieval_temperature" in params:
        return float(params["retrieval_temperature"])
    if "tau" in params:
        return float(params["tau"])
    return float(defaults.get("retrieval_temperature", DEFAULT_GRADNORM_RETRIEVAL_TEMPERATURE))


def run_gradnorm_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> GradNormOutputs:
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger("gradnorm", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "gradnorm",
        bundle.spec.cache_key,
    )

    train_dataset, test_dataset, train_split, test_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="GradNorm",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={test_dataset.name}/{test_split}"
    )

    if train_dataset.num_classes == 0:
        raise ValueError("GradNorm requires datasets with known class names.")

    hyperparameter_defaults = _resolve_gradnorm_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    hyperparameter_profile_source = _resolve_gradnorm_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    cluster_num = int(params.get("n_clusters", test_dataset.num_classes))
    filter_cluster_num, filter_cluster_num_source = _resolve_gradnorm_filter_cluster_num(
        params,
        hyperparameter_defaults,
        train_dataset.name,
        cluster_num,
    )

    image_batch_size = int(params.get("image_batch_size", 256))
    text_batch_size = int(params.get("text_batch_size", 2048))
    retrieval_batch_size = int(params.get("retrieval_batch_size", 8192))
    selected_nouns_per_center = int(params.get("selected_nouns_per_center", params.get("topK", 5)))
    filter_temperature = _resolve_gradnorm_filter_temperature(params, hyperparameter_defaults)
    filter_p = float(params.get("filter_p", params.get("p", 2.0)))
    retrieval_temperature = _resolve_gradnorm_retrieval_temperature(params, hyperparameter_defaults)
    candidate_noun_limit = params.get("candidate_noun_limit")
    random_state = params.get("random_state", params.get("seed"))

    if cluster_num <= 0:
        raise ValueError("GradNorm requires n_clusters > 0.")
    if filter_cluster_num <= 0:
        raise ValueError("GradNorm requires filter_cluster_num > 0.")
    if selected_nouns_per_center <= 0:
        raise ValueError("GradNorm requires selected_nouns_per_center > 0.")
    if filter_temperature <= 0:
        raise ValueError("GradNorm requires filter_temperature > 0.")
    if filter_p <= 0:
        raise ValueError("GradNorm requires filter_p > 0.")
    if retrieval_temperature <= 0:
        raise ValueError("GradNorm requires retrieval_temperature > 0.")

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

    wordnet_csv = _resolve_tac_wordnet_csv(params)
    with progress.stage(f"Encoding WordNet noun pool from {wordnet_csv}"):
        candidate_nouns, noun_embeddings = _encode_wordnet_nouns(
            csv_path=wordnet_csv,
            bundle=bundle,
            batch_size=text_batch_size,
            device=device,
            cache_dir=output_dir,
            candidate_noun_limit=candidate_noun_limit,
            force_recompute=bool(params.get("force_recompute_nouns", False)),
        )
    noun_limit_token = "all" if candidate_noun_limit in {None, "", "none"} else candidate_noun_limit
    selection_cache_key = cache_key(
        image_dataset_cache_key(train_dataset),
        bundle.spec.cache_key,
        wordnet_source_token(wordnet_csv),
        FAISS_KMEANS_CACHE_TOKEN,
        f"limit{noun_limit_token}",
        f"filter{filter_cluster_num}",
        f"nouns{selected_nouns_per_center}",
        f"temp{filter_temperature}",
        f"p{filter_p}",
        f"seed{random_state}",
    )
    eval_retrieval_cache_key = cache_key(
        image_dataset_cache_key(test_dataset),
        bundle.spec.cache_key,
        "source",
        selection_cache_key,
        f"retrieval{retrieval_temperature}",
    )
    efficiency_measurement = start_train_eval_measurement(device)
    with progress.stage(f"Selecting GradNorm nouns with {filter_cluster_num} proxy cluster(s)"):
        selected_noun_embeddings, selected_noun_indices = _select_gradnorm_nouns(
            image_features=train_image_features,
            noun_embeddings=noun_embeddings,
            cluster_num=filter_cluster_num,
            nouns_per_center=selected_nouns_per_center,
            temperature=filter_temperature,
            p=filter_p,
            device=device,
            cache_dir=output_dir,
            cache_key=selection_cache_key,
            random_state=None if random_state is None else int(random_state),
        )
    with progress.stage("Retrieving text counterparts for evaluation images"):
        retrieved_test = _retrieve_text_counterparts(
            image_features=test_image_features,
            noun_embeddings=selected_noun_embeddings,
            temperature=retrieval_temperature,
            batch_size=retrieval_batch_size,
            device=device,
            cache_path=cache_file_path(output_dir, f"{eval_retrieval_cache_key}__retrieved", ".npy"),
        )
    concat_embedding = np.concatenate([test_image_features, retrieved_test], axis=1)
    with progress.stage(f"Running final K-Means with {cluster_num} cluster(s)"):
        predictions = _run_kmeans(
            concat_embedding,
            cluster_num=cluster_num,
            n_iter=int(params.get("kmeans_niter", 300)),
            n_redo=int(params.get("kmeans_nredo", 20)),
            random_state=None if random_state is None else int(random_state),
        )
    efficiency_measurement.stop()

    return GradNormOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=test_labels.tolist() if test_labels is not None else None,
        evaluation_split=test_split,
        metadata={
            "variant": "gradnorm_no_train",
            "dataset": train_dataset.name,
            "source_dataset": train_dataset.name,
            "evaluation_dataset": test_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": test_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "wordnet_csv": str(wordnet_csv),
            "cluster_num": cluster_num,
            "filter_cluster_num": filter_cluster_num,
            "filter_cluster_num_source": filter_cluster_num_source,
            "selected_nouns_per_center": selected_nouns_per_center,
            "filter_temperature": filter_temperature,
            "filter_p": filter_p,
            "retrieval_temperature": retrieval_temperature,
            "selected_noun_count": int(selected_noun_embeddings.shape[0]),
            "selected_noun_examples": [candidate_nouns[index] for index in selected_noun_indices[:10]],
            "train_image_count": int(train_image_features.shape[0]),
            "test_image_count": int(test_image_features.shape[0]),
            "train_labels_available": train_labels is not None,
        },
    )


def _select_gradnorm_nouns(
    image_features: np.ndarray,
    noun_embeddings: np.ndarray,
    cluster_num: int,
    nouns_per_center: int,
    temperature: float,
    p: float,
    device,
    cache_dir: Path,
    cache_key: str,
    random_state: int | None,
) -> tuple[np.ndarray, list[int]]:
    cache_path = cache_file_path(cache_dir, f"{cache_key}__selected_nouns", ".npz")
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return payload["embeddings"], [int(index) for index in payload["indices"].tolist()]

    torch = require_module("torch", "pip install torch")
    normalized_images = _normalize_rows(image_features.astype("float32"))
    normalized_nouns = _normalize_rows(noun_embeddings.astype("float32"))
    assignments = _run_kmeans(
        normalized_images,
        cluster_num=cluster_num,
        n_iter=300,
        n_redo=10,
        random_state=random_state,
    )

    image_tensor = torch.from_numpy(normalized_images).to(device=device, dtype=torch.float32)
    noun_tensor = torch.from_numpy(normalized_nouns).to(device=device, dtype=torch.float32)
    image_centers = torch.zeros((cluster_num, normalized_images.shape[1]), dtype=torch.float32, device=device)
    for index in range(cluster_num):
        members = image_tensor[assignments == index]
        if members.numel() == 0:
            continue
        image_centers[index] = members.mean(dim=0)
    image_centers = torch.nn.functional.normalize(image_centers, dim=1)

    similarity = torch.matmul(image_centers, noun_tensor.t()) / temperature
    softmax_nouns = torch.softmax(similarity, dim=0).float().cpu()
    class_pred = torch.argmax(softmax_nouns, dim=0).long()
    nouns_cpu = noun_tensor.cpu()
    selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)

    for index in range(cluster_num):
        if int((class_pred == index).sum()) == 0:
            continue
        class_index = torch.where(class_pred == index)[0]
        cluster_nouns = nouns_cpu[class_index]
        softmax_class = softmax_nouns[:, class_index].t()
        confidence = softmax_class.max(dim=1)[0]
        score_1 = torch.sum(torch.pow(softmax_class, p), dim=1)
        score_1 = score_1 + torch.pow(1 - confidence, p) - torch.pow(confidence, p)
        score_2 = torch.sum(torch.pow(torch.abs(cluster_nouns), p), dim=1)
        score = torch.pow(score_1 * score_2, 1 / p)
        rank = torch.argsort(-score, descending=True)
        selected_idx[class_index[rank[:nouns_per_center]]] = True

    selected_indices = torch.where(selected_idx)[0].cpu().tolist()
    selected_embeddings = normalized_nouns[selected_indices]
    np.savez_compressed(cache_path, embeddings=selected_embeddings, indices=np.asarray(selected_indices, dtype=np.int64))
    return selected_embeddings, selected_indices
