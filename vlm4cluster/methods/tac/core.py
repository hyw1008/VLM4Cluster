from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.config import DatasetConfig
from vlm4cluster.datasets import load_image_dataset
from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label
from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_file, load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.checkpointing import save_checkpoint_enabled
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
from vlm4cluster.models import OpenCLIPBundle, load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import FAISS_KMEANS_CACHE_TOKEN, run_faiss_kmeans, search_topk
from vlm4cluster.utils.progress import get_progress_logger

from vlm4cluster.methods.tac.losses import TACDistillLoss, consistency_loss, entropy_loss

DOMAIN_SHIFT_TARGET_SPLITS: dict[str, str] = {
    "imageneta": "test",
    "imagenetsketch": "test",
    "imagenetr": "test",
    "imagenetv2": "test",
    "imagenetc": "test",
}

TAC_OFFICIAL_SEMANTIC_CLUSTER_DEFAULTS: dict[str, int] = {
    "cifar10": 167,
    "cifar20": 167,
    "stl10": 17,
    "imagenet10": 43,
    "imagenetdogs": 65,
    "dtd": 141,
    "ucf101": 303,
    "imagenet": 4271,
}

TAC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

TAC_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "cifar10", "train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "cifar20", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "cifar100", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "cifar100", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "stl10", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "stl10", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenet10", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "imagenet10", "train"): {"retrieval_temperature": 0.04},
    ("laion400m", "vitb32", "imagenetdogs", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenetdogs", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "dtd", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "dtd", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "ucf101", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "ucf101", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenet", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenet", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "places365standard", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "places365standard", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "aircraft", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb32", "aircraft", "train"): {
        "retrieval_temperature": 0.03,
        "distill_temperature": 10.0,
    },
    ("laion400m", "vitb32", "cars", "no_train"): {"retrieval_temperature": 0.09},
    ("laion400m", "vitb32", "cars", "train"): {
        "retrieval_temperature": 0.04,
        "distill_temperature": 9.0,
    },
    ("laion400m", "vitb32", "flowers", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb32", "flowers", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "food", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "food", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "pets", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "pets", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "cifar10", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "cifar10", "train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "cifar20", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "cifar20", "train"): {"retrieval_temperature": 0.008},
    ("laion400m", "vitb16", "cifar100", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "cifar100", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "stl10", "no_train"): {"retrieval_temperature": 0.1},
    ("laion400m", "vitb16", "stl10", "train"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitb16", "imagenet10", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "imagenet10", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "imagenetdogs", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "imagenetdogs", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "dtd", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "dtd", "train"): {
        "retrieval_temperature": 0.05,
        "distill_temperature": 2.0,
    },
    ("laion400m", "vitb16", "ucf101", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb16", "ucf101", "train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb16", "imagenet", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "imagenet", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "places365standard", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "places365standard", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "aircraft", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb16", "aircraft", "train"): {
        "retrieval_temperature": 0.03,
        "distill_temperature": 3.0,
    },
    ("laion400m", "vitb16", "cars", "no_train"): {"retrieval_temperature": 0.04},
    ("laion400m", "vitb16", "cars", "train"): {
        "retrieval_temperature": 0.05,
        "distill_temperature": 7.0,
    },
    ("laion400m", "vitb16", "flowers", "no_train"): {"retrieval_temperature": 0.04},
    ("laion400m", "vitb16", "flowers", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "food", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "food", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "pets", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb16", "pets", "train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "cifar10", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "cifar10", "train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "cifar20", "no_train"): {"retrieval_temperature": 0.08},
    ("laion400m", "vitl14", "cifar20", "train"): {"retrieval_temperature": 0.09},
    ("laion400m", "vitl14", "cifar100", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "cifar100", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "stl10", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "stl10", "train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "imagenet10", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "imagenet10", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "imagenetdogs", "no_train"): {"retrieval_temperature": 0.04},
    ("laion400m", "vitl14", "imagenetdogs", "train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "dtd", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "dtd", "train"): {
        "retrieval_temperature": 0.02,
        "distill_temperature": 1.0,
    },
    ("laion400m", "vitl14", "ucf101", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "ucf101", "train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "imagenet", "no_train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "imagenet", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "places365standard", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "places365standard", "train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "aircraft", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "aircraft", "train"): {
        "retrieval_temperature": 0.05,
        "distill_temperature": 3.0,
    },
    ("laion400m", "vitl14", "cars", "no_train"): {"retrieval_temperature": 0.07},
    ("laion400m", "vitl14", "cars", "train"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "flowers", "no_train"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitl14", "flowers", "train"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "food", "no_train"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "food", "train"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "pets", "no_train"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "pets", "train"): {"retrieval_temperature": 0.02},
}


@dataclass(slots=True)
class TACOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class TACClusterHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_dim: int, num_clusters: int) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.cluster_head_text = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.BatchNorm1d(in_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim, num_clusters),
            torch.nn.Softmax(dim=1),
        )
        self.cluster_head_image = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.BatchNorm1d(in_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim, num_clusters),
            torch.nn.Softmax(dim=1),
        )
        torch.nn.init.trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        torch.nn.init.trunc_normal_(self.cluster_head_text[3].weight, std=0.02)
        torch.nn.init.trunc_normal_(self.cluster_head_image[0].weight, std=0.02)
        torch.nn.init.trunc_normal_(self.cluster_head_image[3].weight, std=0.02)

    def forward(self, text_features, image_features):
        return self.cluster_head_text(text_features), self.cluster_head_image(image_features)


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


class NeighborPairsDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        text_features,
        image_features,
        text_neighbor_indices: np.ndarray,
        image_neighbor_indices: np.ndarray,
        seed: int | None = None,
    ) -> None:
        self.text_features = text_features
        self.image_features = image_features
        self.text_neighbor_indices = text_neighbor_indices
        self.image_neighbor_indices = image_neighbor_indices
        self.np_random = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.text_features.size(0)

    def __getitem__(self, index: int):
        text_neighbor = int(self.np_random.choice(self.text_neighbor_indices[index]))
        image_neighbor = int(self.np_random.choice(self.image_neighbor_indices[index]))
        return (
            self.text_features[index],
            self.image_features[index],
            self.text_features[text_neighbor],
            self.image_features[image_neighbor],
        )


def default_split_mapping(dataset_name: str) -> tuple[str, str]:
    mapping = {
        "cifar10": ("train", "test"),
        "cifar20": ("train", "test"),
        "cifar100": ("train", "test"),
        "stl10": ("train", "test"),
        "imagenet10": ("train", "val"),
        "imagenetdogs": ("train", "val"),
        "dtd": ("trainval", "test"),
        "ucf101": ("train", "val"),
        "imagenet": ("train", "val"),
        "places365standard": ("train", "val"),
        "aircraft": ("train", "test"),
        "cars": ("train", "test"),
        "flowers": ("train", "test"),
        "food": ("train", "test"),
        "pets": ("train", "test"),
    }
    normalized = _normalize_dataset_name(dataset_name)
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"TAC does not define default train/test splits for dataset '{dataset_name}'.") from exc


def default_evaluation_split(dataset_name: str) -> str:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized in DOMAIN_SHIFT_TARGET_SPLITS:
        return DOMAIN_SHIFT_TARGET_SPLITS[normalized]
    return default_split_mapping(dataset_name)[1]


def _normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _normalize_tac_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_tac_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in TAC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _tac_variant_profile_key(train_cluster_heads: bool) -> str:
    return "train" if train_cluster_heads else "no_train"


def _resolve_tac_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
    *,
    train_cluster_heads: bool,
) -> dict[str, Any]:
    profile_key = (
        _normalize_tac_profile_token(openclip_pretraining),
        _normalize_tac_profile_token(openclip_backbone),
        _normalize_tac_profile_dataset(dataset_name),
        _tac_variant_profile_key(train_cluster_heads),
    )
    return dict(TAC_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_tac_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
    *,
    train_cluster_heads: bool,
) -> str | None:
    profile_key = (
        _normalize_tac_profile_token(openclip_pretraining),
        _normalize_tac_profile_token(openclip_backbone),
        _normalize_tac_profile_dataset(dataset_name),
        _tac_variant_profile_key(train_cluster_heads),
    )
    if profile_key not in TAC_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_tac_retrieval_temperature(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "retrieval_temperature" in params:
        return float(params["retrieval_temperature"])
    if "tau" in params:
        return float(params["tau"])
    return float(defaults.get("retrieval_temperature", 0.005))


def _is_imagenet_variant_dataset(name: str) -> bool:
    return _normalize_dataset_name(name) in DOMAIN_SHIFT_TARGET_SPLITS


def _build_named_split_dataset(inputs: MethodInputs, dataset_name: str, split: str, max_samples: Any) -> LoadedImageDataset:
    dataset_download = (
        inputs.image_dataset_config.download
        if _normalize_dataset_name(dataset_name) == _normalize_dataset_name(inputs.image_dataset_config.name)
        else None
    )
    config = replace(
        inputs.image_dataset_config,
        name=dataset_name,
        split=split,
        download=dataset_download,
        transform_preset="none",
        max_samples=max_samples if max_samples is not None else inputs.image_dataset_config.max_samples,
    )
    return load_image_dataset(config)


def _resolve_domain_shift_datasets(
    inputs: MethodInputs,
    params: dict[str, Any],
    *,
    method_name: str,
) -> tuple[LoadedImageDataset, LoadedImageDataset, str, str, bool]:
    source_dataset_name = str(params.get("source_dataset_name", inputs.image_dataset.name))
    target_dataset_name = params.get("target_dataset_name", params.get("target_dataset"))

    train_split_default, test_split_default = default_split_mapping(source_dataset_name)
    train_split = str(params.get("train_split", train_split_default))
    train_dataset = _build_named_split_dataset(inputs, source_dataset_name, train_split, params.get("max_train_samples"))

    if target_dataset_name is None:
        eval_dataset_name = source_dataset_name
        eval_split_default = test_split_default
        domain_shift = False
    else:
        eval_dataset_name = str(target_dataset_name)
        eval_split_default = default_evaluation_split(eval_dataset_name)
        domain_shift = True
        if _is_imagenet_variant_dataset(eval_dataset_name) and _normalize_dataset_name(source_dataset_name) != "imagenet":
            raise ValueError(
                f"{method_name} domain-shift protocol requires ImageNet as the source dataset when the target is "
                f"an ImageNet variant, got source='{source_dataset_name}' and target='{eval_dataset_name}'."
            )

    eval_split = str(params.get("target_test_split", params.get("test_split", eval_split_default)))
    eval_dataset = _build_named_split_dataset(inputs, eval_dataset_name, eval_split, params.get("max_test_samples"))
    return train_dataset, eval_dataset, train_split, eval_split, domain_shift


def _resolve_tac_cluster_num(
    params: dict[str, Any],
    train_dataset: LoadedImageDataset,
) -> tuple[int, str]:
    explicit = params.get("n_clusters")
    if explicit is not None:
        return int(explicit), "explicit"
    if train_dataset.num_classes <= 0:
        raise ValueError("TAC requires source train datasets with known class names to infer n_clusters.")
    return int(train_dataset.num_classes), "source_train_dataset"


def run_tac_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> TACOutputs:
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger("tac", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "tac",
        bundle.spec.cache_key,
    )

    train_dataset, test_dataset, train_split, test_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="TAC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={test_dataset.name}/{test_split}"
    )

    if train_dataset.num_classes == 0:
        raise ValueError("TAC requires datasets with known class names.")

    train_cluster_heads = bool(params.get("train_cluster_heads", True))
    cluster_num, cluster_num_source = _resolve_tac_cluster_num(
        params,
        train_dataset,
    )
    progress.log(f"Resolved TAC n_clusters={cluster_num} ({cluster_num_source})")
    hyperparameter_defaults = _resolve_tac_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
        train_cluster_heads=train_cluster_heads,
    )
    hyperparameter_profile_source = _resolve_tac_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
        train_cluster_heads=train_cluster_heads,
    )
    image_batch_size = int(params.get("image_batch_size", 256))
    text_batch_size = int(params.get("text_batch_size", 2048))
    retrieval_batch_size = int(params.get("retrieval_batch_size", 8192))
    selected_nouns_per_center = int(params.get("selected_nouns_per_center", 5))
    retrieval_temperature = _resolve_tac_retrieval_temperature(params, hyperparameter_defaults)
    semantic_cluster_size = int(params.get("semantic_cluster_size", 300))
    candidate_noun_limit = params.get("candidate_noun_limit")
    semantic_clusters = params.get("semantic_clusters")
    random_state = params.get("random_state", params.get("seed"))
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)

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

    effective_semantic_clusters = _resolve_semantic_cluster_count(
        dataset_name=train_dataset.name,
        train_size=train_image_features.shape[0],
        cluster_num=cluster_num,
        semantic_cluster_size=semantic_cluster_size,
        explicit_value=semantic_clusters,
    )
    noun_limit_token = "all" if candidate_noun_limit in {None, "", "none"} else candidate_noun_limit
    train_feature_cache_key = cache_key(image_dataset_cache_key(train_dataset), bundle.spec.cache_key)
    selection_cache_key = cache_key(
        train_feature_cache_key,
        wordnet_source_token(wordnet_csv),
        FAISS_KMEANS_CACHE_TOKEN,
        f"limit{noun_limit_token}",
        f"semantic{effective_semantic_clusters}",
        f"nouns{selected_nouns_per_center}",
        f"seed{random_state}",
    )
    train_retrieval_cache_key = cache_key(selection_cache_key, f"retrieval{retrieval_temperature}")
    eval_retrieval_cache_key = cache_key(
        image_dataset_cache_key(test_dataset),
        bundle.spec.cache_key,
        "source",
        selection_cache_key,
        f"retrieval{retrieval_temperature}",
    )
    no_train_efficiency_measurement = (
        start_train_eval_measurement(device) if not train_cluster_heads else None
    )
    with progress.stage(
        f"Selecting discriminative nouns with {effective_semantic_clusters} semantic cluster(s)"
    ):
        selected_noun_embeddings, selected_noun_indices = _select_discriminative_nouns(
            image_features=train_image_features,
            noun_embeddings=noun_embeddings,
            cluster_num=effective_semantic_clusters,
            nouns_per_center=selected_nouns_per_center,
            cache_dir=output_dir,
            cache_key=selection_cache_key,
            random_state=None if random_state is None else int(random_state),
        )

    with progress.stage("Retrieving text counterparts for train images"):
        retrieved_train = _retrieve_text_counterparts(
            image_features=train_image_features,
            noun_embeddings=selected_noun_embeddings,
            temperature=retrieval_temperature,
            batch_size=retrieval_batch_size,
            device=device,
            cache_path=cache_file_path(output_dir, f"{train_retrieval_cache_key}__retrieved", ".npy"),
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

    if not train_cluster_heads:
        progress.log("Running TAC no-train concat K-Means variant")
        concat_embedding = np.concatenate([test_image_features, retrieved_test], axis=1)
        predictions = _run_kmeans(
            concat_embedding,
            cluster_num=cluster_num,
            n_iter=int(params.get("kmeans_niter", 300)),
            n_redo=int(params.get("kmeans_nredo", 20)),
            random_state=None if random_state is None else int(random_state),
        )
        no_train_efficiency_measurement.stop()
        return TACOutputs(
            predictions=predictions.tolist(),
            evaluation_labels=test_labels.tolist() if test_labels is not None else None,
            evaluation_split=test_split,
            metadata={
                "variant": "tac_no_train",
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
                "cluster_num_source": cluster_num_source,
                "retrieval_temperature": retrieval_temperature,
                "hyperparameter_default_profile": hyperparameter_profile_source,
                "semantic_clusters": effective_semantic_clusters,
                "semantic_clusters_source": _resolve_semantic_cluster_source(train_dataset.name, semantic_clusters),
                "selected_noun_count": int(selected_noun_embeddings.shape[0]),
                "selected_noun_examples": [candidate_nouns[index] for index in selected_noun_indices[:10]],
            },
        )

    train_params = _resolve_training_params(
        cluster_num=cluster_num,
        params=params,
        hyperparameter_defaults=hyperparameter_defaults,
    )
    checkpoint_name = f"{cache_key(train_feature_cache_key, train_retrieval_cache_key, f'k{cluster_num}')}__tac_final.pt"
    checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    checkpoint_loaded = False
    if load_checkpoint_enabled(params, train_dataset.name) and checkpoint_path.exists():
        progress.log(f"Loading TAC checkpoint and skipping head training: {checkpoint_path}")
        predictions, training_history = _infer_tac_checkpoint(
            test_image_features=test_image_features,
            cluster_num=cluster_num,
            batch_size=train_params["batch_size"],
            device=device,
            checkpoint_path=checkpoint_path,
        )
        effective_train_batch_size = train_params["batch_size"]
        checkpoint_loaded = True
    else:
        save_path = checkpoint_file(output_dir, checkpoint_name, create=True) if save_checkpoint else None
        predictions, training_history, effective_train_batch_size = _train_cluster_heads(
            train_image_features=train_image_features,
            train_text_features=retrieved_train,
            test_image_features=test_image_features,
            cluster_num=cluster_num,
            device=device,
            output_dir=output_dir,
            image_neighbor_cache_key=train_feature_cache_key,
            text_neighbor_cache_key=train_retrieval_cache_key,
            checkpoint_path=save_path,
            progress=progress,
            **train_params,
        )
        checkpoint_path = save_path or checkpoint_path
    return TACOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=test_labels.tolist() if test_labels is not None else None,
        evaluation_split=test_split,
        metadata={
            "variant": "tac",
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
            "cluster_num_source": cluster_num_source,
            "retrieval_temperature": retrieval_temperature,
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "semantic_clusters": effective_semantic_clusters,
            "semantic_clusters_source": _resolve_semantic_cluster_source(train_dataset.name, semantic_clusters),
            "selected_noun_count": int(selected_noun_embeddings.shape[0]),
            "selected_noun_examples": [candidate_nouns[index] for index in selected_noun_indices[:10]],
            "train_image_count": int(train_image_features.shape[0]),
            "test_image_count": int(test_image_features.shape[0]),
            "train_epochs": train_params["epochs"],
            "train_batch_size": effective_train_batch_size,
            "requested_train_batch_size": train_params["batch_size"],
            "distill_temperature": train_params["distill_temperature"],
            "save_checkpoint": save_checkpoint,
            "checkpoint_path": str(checkpoint_path) if (checkpoint_loaded or save_checkpoint) else None,
            "load_checkpoint": load_checkpoint_enabled(params, train_dataset.name),
            "checkpoint_loaded": checkpoint_loaded,
            "train_labels_available": train_labels is not None,
            "training_history": training_history,
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


def _resolve_tac_wordnet_csv(params: dict[str, Any]) -> Path:
    explicit_path = params.get("wordnet_csv") or params.get("wordnet_path")
    if explicit_path is not None:
        candidate = Path(str(explicit_path)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"TAC WordNet CSV not found: {candidate}")

    default_candidate = Path.cwd() / "data" / "WordNetNouns.csv"
    if default_candidate.exists():
        return default_candidate.resolve()

    raise FileNotFoundError(
        "Language-assisted methods require the shared WordNetNouns.csv resource. "
        "Please provide method.params.wordnet_csv or place the file at: "
        f"{default_candidate}"
    )


def _load_tac_wordnet_nouns(csv_path: Path, limit: int | None) -> list[str]:
    nouns: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "word" not in reader.fieldnames:
            raise ValueError(f"Invalid TAC WordNet CSV format: {csv_path}")
        for row in reader:
            word = str(row["word"]).strip()
            if word:
                nouns.append(word)
    if limit is not None:
        return nouns[:limit]
    return nouns


def _encode_wordnet_nouns(
    csv_path: Path,
    bundle: OpenCLIPBundle,
    batch_size: int,
    device,
    cache_dir: Path,
    candidate_noun_limit: Any,
    force_recompute: bool,
) -> tuple[list[str], np.ndarray]:
    limit_value = None if candidate_noun_limit in {None, "", "none"} else int(candidate_noun_limit)
    all_nouns = _load_tac_wordnet_nouns(csv_path, None)
    _, _, ensemble_embeddings = load_or_compute_text_prompt_bank(
        csv_path=csv_path,
        nouns=all_nouns,
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
    if limit_value is None:
        return all_nouns, ensemble_embeddings
    return all_nouns[:limit_value], ensemble_embeddings[:limit_value]


def _resolve_semantic_cluster_count(
    dataset_name: str,
    train_size: int,
    cluster_num: int,
    semantic_cluster_size: int,
    explicit_value: Any,
) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    official_default = TAC_OFFICIAL_SEMANTIC_CLUSTER_DEFAULTS.get(_normalize_dataset_name(dataset_name))
    if official_default is not None:
        return max(int(official_default), cluster_num)
    average_cluster_size = train_size / cluster_num
    if average_cluster_size < semantic_cluster_size:
        return max(cluster_num * 3, cluster_num)
    return max(int(round(train_size / semantic_cluster_size)), cluster_num)


def _resolve_semantic_cluster_source(dataset_name: str, explicit_value: Any) -> str:
    if explicit_value is not None:
        return "explicit"
    if _normalize_dataset_name(dataset_name) in TAC_OFFICIAL_SEMANTIC_CLUSTER_DEFAULTS:
        return "official_default"
    return "size_heuristic"


def _select_discriminative_nouns(
    image_features: np.ndarray,
    noun_embeddings: np.ndarray,
    cluster_num: int,
    nouns_per_center: int,
    cache_dir: Path,
    cache_key: str,
    random_state: int | None,
) -> tuple[np.ndarray, list[int]]:
    cache_path = cache_file_path(cache_dir, f"{cache_key}__selected_nouns", ".npz")
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return payload["embeddings"], [int(index) for index in payload["indices"].tolist()]

    assignments = _run_kmeans(
        image_features,
        cluster_num=cluster_num,
        n_iter=300,
        n_redo=10,
        random_state=random_state,
    )
    centers = []
    for index in range(cluster_num):
        members = image_features[assignments == index]
        if len(members) == 0:
            centers.append(np.zeros((image_features.shape[1],), dtype=np.float32))
        else:
            centers.append(members.mean(axis=0))
    centers = _normalize_rows(np.stack(centers, axis=0))

    torch = require_module("torch", "pip install torch")
    similarity = torch.from_numpy(centers) @ torch.from_numpy(noun_embeddings).t()
    confidence = torch.softmax(similarity, dim=0)
    class_prediction = torch.argmax(confidence, dim=0)

    selected_mask = torch.zeros_like(class_prediction, dtype=torch.bool)
    for center_index in range(cluster_num):
        member_indices = torch.where(class_prediction == center_index)[0]
        if member_indices.numel() == 0:
            continue
        member_confidence = confidence[:, member_indices].max(dim=0)[0]
        ranking = torch.argsort(member_confidence, descending=True)
        keep = member_indices[ranking[:nouns_per_center]]
        selected_mask[keep] = True

    selected_indices = torch.where(selected_mask)[0].cpu().tolist()
    selected_embeddings = noun_embeddings[selected_indices]
    np.savez_compressed(cache_path, embeddings=selected_embeddings, indices=np.asarray(selected_indices, dtype=np.int64))
    return selected_embeddings, selected_indices


def _retrieve_text_counterparts(
    image_features: np.ndarray,
    noun_embeddings: np.ndarray,
    temperature: float,
    batch_size: int,
    device,
    cache_path: Path,
) -> np.ndarray:
    progress = get_progress_logger("tac")
    if cache_path.exists():
        progress.log(f"Loading cached retrieved text counterparts: {cache_path}")
        return np.load(cache_path)

    torch = require_module("torch", "pip install torch")
    noun_tensor = torch.from_numpy(noun_embeddings.astype("float32")).to(device)
    noun_tensor = torch.nn.functional.normalize(noun_tensor, dim=1)
    image_tensor = torch.from_numpy(image_features.astype("float32")).to(device)
    image_tensor = torch.nn.functional.normalize(image_tensor, dim=1)

    retrieved = []
    total_batches = max(1, (image_tensor.size(0) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, image_tensor.size(0), batch_size), start=1):
        batch = image_tensor[start : start + batch_size]
        similarity = batch @ noun_tensor.t()
        similarity = torch.softmax(similarity / temperature, dim=1)
        batch_retrieved = similarity @ noun_tensor
        batch_retrieved = torch.nn.functional.normalize(batch_retrieved, dim=1)
        retrieved.append(batch_retrieved.cpu().numpy().astype("float32"))
        progress.step("Retrieving text counterparts", batch_index, total_batches, noun="batch")
    retrieved_matrix = np.concatenate(retrieved, axis=0)
    np.save(cache_path, retrieved_matrix)
    progress.log(f"Saved retrieved text counterparts: {cache_path}")
    return retrieved_matrix


def _run_kmeans(
    matrix: np.ndarray,
    cluster_num: int,
    n_iter: int,
    n_redo: int,
    random_state: int | None,
) -> np.ndarray:
    normalized = _normalize_rows(matrix.astype("float32"))
    assignments, _ = run_faiss_kmeans(
        normalized,
        n_clusters=cluster_num,
        n_iter=n_iter,
        n_redo=n_redo,
        spherical=True,
        random_state=random_state,
    )
    return assignments


def _compute_neighbors(
    features: np.ndarray,
    topk: int,
    cache_path: Path,
    *,
    progress=None,
    label: str | None = None,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    normalized = _normalize_rows(features.astype("float32"))
    _, indices = search_topk(
        normalized,
        topk=topk + 1,
        faiss_metric="ip",
        sklearn_metric="cosine",
        progress=progress,
        label=label,
    )
    neighbors = indices[:, 1:]

    np.save(cache_path, neighbors.astype(np.int64))
    return neighbors


def _resolve_training_params(
    cluster_num: int,
    params: dict[str, Any],
    hyperparameter_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cluster_num >= 100:
        defaults = {"epochs": 100, "batch_size": 8192, "distill_temperature": 5.0}
    else:
        defaults = {"epochs": 20, "batch_size": 512, "distill_temperature": 0.5}
    if hyperparameter_defaults is not None:
        for key in ("epochs", "batch_size", "distill_temperature"):
            if key in hyperparameter_defaults:
                defaults[key] = hyperparameter_defaults[key]

    return {
        "epochs": int(params.get("epochs", defaults["epochs"])),
        "batch_size": int(params.get("train_batch_size", defaults["batch_size"])),
        "distill_temperature": float(params.get("distill_temperature", defaults["distill_temperature"])),
        "learning_rate": float(params.get("learning_rate", 1e-3)),
        "neighbors_topk": int(params.get("neighbors_topk", 50)),
        "balance_weight": float(params.get("balance_weight", 5.0)),
        "seed": int(params.get("seed", 42)),
    }


def _train_cluster_heads(
    train_image_features: np.ndarray,
    train_text_features: np.ndarray,
    test_image_features: np.ndarray,
    cluster_num: int,
    epochs: int,
    batch_size: int,
    distill_temperature: float,
    learning_rate: float,
    neighbors_topk: int,
    balance_weight: float,
    seed: int,
    device,
    output_dir: Path,
    image_neighbor_cache_key: str,
    text_neighbor_cache_key: str,
    checkpoint_path: Path | None = None,
    progress=None,
) -> tuple[np.ndarray, list[dict[str, float]], int]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = progress or get_progress_logger("tac")

    progress.log(f"Computing TAC nearest-neighbor pairs with topk={neighbors_topk}")
    text_neighbors = _compute_neighbors(
        train_text_features,
        topk=neighbors_topk,
        cache_path=cache_file_path(output_dir, f"{text_neighbor_cache_key}__text_neighbors_topk{neighbors_topk}", ".npy"),
        progress=progress,
        label="TAC text-neighbor search",
    )
    image_neighbors = _compute_neighbors(
        train_image_features,
        topk=neighbors_topk,
        cache_path=cache_file_path(output_dir, f"{image_neighbor_cache_key}__image_neighbors_topk{neighbors_topk}", ".npy"),
        progress=progress,
        label="TAC image-neighbor search",
    )

    train_text_tensor = torch.from_numpy(train_text_features.astype("float32"))
    train_image_tensor = torch.from_numpy(train_image_features.astype("float32"))
    test_image_tensor = torch.from_numpy(test_image_features.astype("float32"))

    train_dataset = NeighborPairsDataset(
        text_features=train_text_tensor,
        image_features=train_image_tensor,
        text_neighbor_indices=text_neighbors,
        image_neighbor_indices=image_neighbors,
        seed=seed,
    )
    test_dataset = data_mod.TensorDataset(test_image_tensor)

    if len(train_dataset) < 2:
        raise ValueError("TAC cluster-head training requires at least two training samples.")

    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size != batch_size:
        progress.log(
            f"Reducing TAC train batch size from {batch_size} to {effective_batch_size} "
            f"for {len(train_dataset)} training sample(s)"
        )

    drop_last = len(train_dataset) > effective_batch_size and len(train_dataset) % effective_batch_size == 1
    train_loader = data_mod.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    test_loader = data_mod.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TACClusterHead(in_dim=train_image_features.shape[1], num_clusters=cluster_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)
    distill_loss = TACDistillLoss(class_num=cluster_num, temperature=distill_temperature).to(device)

    history: list[dict[str, float]] = []
    efficiency_measurement = start_train_eval_measurement(device)
    progress.log(f"Training TAC cluster heads for {epochs} epoch(s)")
    for epoch in range(epochs):
        model.train()
        epoch_distill = 0.0
        epoch_consistency = 0.0
        epoch_entropy = 0.0
        iterations = 0
        for text_anchor, image_anchor, text_neighbor, image_neighbor in train_loader:
            text_anchor = text_anchor.to(device)
            image_anchor = image_anchor.to(device)
            text_neighbor = text_neighbor.to(device)
            image_neighbor = image_neighbor.to(device)

            text_probs, image_probs = model(text_anchor, image_anchor)
            neighbor_text_probs, neighbor_image_probs = model(text_neighbor, image_neighbor)

            loss_distill = distill_loss(image_probs, neighbor_text_probs) + distill_loss(text_probs, neighbor_image_probs)
            loss_consistency = consistency_loss(text_probs, image_probs)
            loss_entropy = entropy_loss(text_probs) + entropy_loss(image_probs)
            loss = loss_distill + loss_consistency - balance_weight * loss_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_distill += float(loss_distill.item())
            epoch_consistency += float(loss_consistency.item())
            epoch_entropy += float(loss_entropy.item())
            iterations += 1

        epoch_record = {
            "epoch": float(epoch + 1),
            "loss_distill": epoch_distill / max(iterations, 1),
            "loss_consistency": epoch_consistency / max(iterations, 1),
            "loss_entropy": epoch_entropy / max(iterations, 1),
        }
        history.append(epoch_record)
        progress.epoch(
            "TAC training",
            epoch + 1,
            epochs,
            metrics={
                "distill": epoch_record["loss_distill"],
                "consistency": epoch_record["loss_consistency"],
                "entropy": epoch_record["loss_entropy"],
            },
        )

    progress.log("Inferring TAC cluster assignments on the evaluation split")
    predictions = _infer_cluster_predictions(model, test_loader, device)
    efficiency_measurement.stop()
    if checkpoint_path is not None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cluster_num": cluster_num,
                "epochs": epochs,
                "history": history,
            },
            checkpoint_path,
        )
    return predictions, history, effective_batch_size


def _infer_tac_checkpoint(
    *,
    test_image_features: np.ndarray,
    cluster_num: int,
    batch_size: int,
    device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    model = TACClusterHead(in_dim=test_image_features.shape[1], num_clusters=cluster_num).to(device)
    model.load_state_dict(payload["model"], strict=True)
    test_tensor = torch.from_numpy(test_image_features.astype("float32"))
    test_loader = data_mod.DataLoader(data_mod.TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)
    predictions = _infer_cluster_predictions(model, test_loader, device)
    history = payload.get("history", [])
    return predictions, history if isinstance(history, list) else []


def _infer_cluster_predictions(model, dataloader, device) -> np.ndarray:
    torch = require_module("torch", "pip install torch")
    model.eval()
    predictions = []
    with torch.no_grad():
        for (image_features,) in dataloader:
            image_features = image_features.to(device)
            _, image_probs = model(image_features, image_features)
            predictions.append(torch.argmax(image_probs, dim=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)
