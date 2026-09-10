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
from vlm4cluster.methods.checkpointing import checkpoint_file, load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.checkpointing import save_checkpoint_enabled
from vlm4cluster.methods.cluster_defaults import resolve_source_default_cluster_num
from vlm4cluster.methods.sac.losses import (
    SACDataContrastiveLoss,
    bidirectional_alignment_loss,
    entropy_loss,
)
from vlm4cluster.methods.tac.core import (
    _compute_neighbors,
    _encode_image_dataset,
    _encode_wordnet_nouns,
    _normalize_dataset_name,
    _resolve_domain_shift_datasets,
    _resolve_semantic_cluster_count,
    _resolve_semantic_cluster_source,
    _resolve_tac_wordnet_csv,
    _select_discriminative_nouns,
)
from vlm4cluster.models import load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.progress import get_progress_logger


SAC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

SAC_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "imagenet"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "aircraft"): {"retrieval_temperature": 0.06},
    ("laion400m", "vitb32", "cars"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitb32", "flowers"): {
        "retrieval_temperature": 0.01,
        "contrastive_temperature": 2.0,
    },
    ("laion400m", "vitb32", "food"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "cifar10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "cifar100"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitb16", "imagenet"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "aircraft"): {
        "retrieval_temperature": 0.06,
        "contrastive_temperature": 2.0,
    },
    ("laion400m", "vitb16", "cars"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitb16", "flowers"): {
        "retrieval_temperature": 0.001,
        "contrastive_temperature": 2.0,
    },
    ("laion400m", "vitb16", "food"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "cifar20"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "dtd"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "imagenet"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "aircraft"): {"retrieval_temperature": 0.06},
    ("laion400m", "vitl14", "cars"): {"retrieval_temperature": 0.05},
    ("laion400m", "vitl14", "food"): {"retrieval_temperature": 0.01},
}


@dataclass(slots=True)
class SACOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class SACClusterHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_dim: int, text_in_dim: int, num_clusters: int) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.in_dim = in_dim
        self.text_in_dim = text_in_dim
        self.text_proj = torch.nn.Linear(text_in_dim, in_dim) if text_in_dim != in_dim else torch.nn.Identity()
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
        for module in (
            self.cluster_head_text[0],
            self.cluster_head_text[3],
            self.cluster_head_image[0],
            self.cluster_head_image[3],
        ):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, text_features, image_features):
        if text_features.size(1) != self.in_dim:
            text_features = self.text_proj(text_features)
        return self.cluster_head_text(text_features), self.cluster_head_image(image_features)


class SACNeighborDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        text_features,
        image_features,
        text_neighbor_indices: np.ndarray,
        image_neighbor_indices: np.ndarray,
        seed: int | None = None,
    ) -> None:
        if text_features.size(0) != image_features.size(0):
            raise ValueError("SAC requires one text counterpart for each train image.")
        self.text_features = text_features
        self.image_features = image_features
        self.text_neighbor_indices = text_neighbor_indices
        self.image_neighbor_indices = image_neighbor_indices
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.text_features.size(0)

    def __getitem__(self, index: int):
        text_neighbor_index = int(self.rng.choice(self.text_neighbor_indices[index]))
        image_neighbor_index = int(self.rng.choice(self.image_neighbor_indices[index]))
        image_to_text_index = int(self.rng.choice(self.image_neighbor_indices[index]))
        text_to_image_index = int(self.rng.choice(self.text_neighbor_indices[index]))
        return (
            self.text_features[index],
            self.image_features[index],
            self.text_features[text_neighbor_index],
            self.image_features[image_neighbor_index],
            self.text_features[image_to_text_index],
            self.image_features[text_to_image_index],
        )


def _normalize_sac_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_sac_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in SAC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _resolve_sac_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_sac_profile_token(openclip_pretraining),
        _normalize_sac_profile_token(openclip_backbone),
        _normalize_sac_profile_dataset(dataset_name),
    )
    return dict(SAC_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_sac_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    profile_key = (
        _normalize_sac_profile_token(openclip_pretraining),
        _normalize_sac_profile_token(openclip_backbone),
        _normalize_sac_profile_dataset(dataset_name),
    )
    if profile_key not in SAC_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_sac_retrieval_temperature(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "retrieval_temperature" in params:
        return float(params["retrieval_temperature"])
    if "tau" in params:
        return float(params["tau"])
    return float(defaults.get("retrieval_temperature", 0.005))


def run_sac_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> SACOutputs:
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger("sac", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "sac",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="SAC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if train_dataset.num_classes == 0:
        raise ValueError("SAC requires datasets with known class names on the source train split.")

    cluster_num, cluster_num_source = resolve_source_default_cluster_num(
        params,
        train_dataset,
        method_name="SAC",
    )
    if cluster_num <= 0:
        raise ValueError("SAC requires n_clusters > 0.")

    hyperparameter_defaults = _resolve_sac_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    hyperparameter_profile_source = _resolve_sac_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    image_batch_size = int(params.get("image_batch_size", 256))
    text_batch_size = int(params.get("text_batch_size", 2048))
    retrieval_batch_size = int(params.get("retrieval_batch_size", 8192))
    selected_nouns_per_center = int(params.get("selected_nouns_per_center", 5))
    retrieval_temperature = _resolve_sac_retrieval_temperature(params, hyperparameter_defaults)
    semantic_cluster_size = int(params.get("semantic_cluster_size", 300))
    candidate_noun_limit = params.get("candidate_noun_limit")
    semantic_clusters = params.get("semantic_clusters")
    random_state = params.get("random_state", params.get("seed"))

    with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
        train_image_features, train_labels = _encode_image_dataset(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    with progress.stage(f"Encoding evaluation image split '{eval_dataset.name}/{eval_split}'"):
        eval_image_features, eval_labels = _encode_image_dataset(
            dataset=eval_dataset,
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
        f"limit{noun_limit_token}",
        f"semantic{effective_semantic_clusters}",
        f"nouns{selected_nouns_per_center}",
        f"seed{random_state}",
    )
    train_retrieval_cache_key = cache_key(selection_cache_key, f"retrieval{retrieval_temperature}")
    eval_retrieval_cache_key = cache_key(
        image_dataset_cache_key(eval_dataset),
        bundle.spec.cache_key,
        "source",
        selection_cache_key,
        f"retrieval{retrieval_temperature}",
    )

    with progress.stage(f"Selecting TAC-style discriminative nouns with {effective_semantic_clusters} semantic cluster(s)"):
        selected_noun_embeddings, selected_noun_indices = _select_discriminative_nouns(
            image_features=train_image_features,
            noun_embeddings=noun_embeddings,
            cluster_num=effective_semantic_clusters,
            nouns_per_center=selected_nouns_per_center,
            cache_dir=output_dir,
            cache_key=selection_cache_key,
            random_state=None if random_state is None else int(random_state),
        )
    if selected_noun_embeddings.shape[0] == 0:
        raise ValueError("SAC TAC-style noun filtering selected zero nouns.")

    with progress.stage("Retrieving TAC-style text counterparts for train images"):
        retrieved_train = _retrieve_text_counterparts(
            image_features=train_image_features,
            noun_embeddings=selected_noun_embeddings,
            temperature=retrieval_temperature,
            batch_size=retrieval_batch_size,
            device=device,
            cache_path=cache_file_path(output_dir, f"{train_retrieval_cache_key}__retrieved", ".npy"),
            progress=progress,
        )
    with progress.stage("Retrieving TAC-style text counterparts for evaluation images"):
        _retrieve_text_counterparts(
            image_features=eval_image_features,
            noun_embeddings=selected_noun_embeddings,
            temperature=retrieval_temperature,
            batch_size=retrieval_batch_size,
            device=device,
            cache_path=cache_file_path(output_dir, f"{eval_retrieval_cache_key}__retrieved", ".npy"),
            progress=progress,
    )

    train_params = _resolve_training_params(
        cluster_num=cluster_num,
        params=params,
        hyperparameter_defaults=hyperparameter_defaults,
    )
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)
    checkpoint_name = f"{cache_key(train_feature_cache_key, train_retrieval_cache_key, f'k{cluster_num}')}__sac_final.pt"
    checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    checkpoint_loaded = False
    if load_checkpoint_enabled(params, train_dataset.name) and checkpoint_path.exists():
        progress.log(f"Loading SAC checkpoint and skipping head training: {checkpoint_path}")
        predictions, training_history = _infer_sac_checkpoint(
            train_text_features=retrieved_train,
            eval_image_features=eval_image_features,
            cluster_num=cluster_num,
            batch_size=train_params["batch_size"],
            device=device,
            checkpoint_path=checkpoint_path,
        )
        effective_train_batch_size = train_params["batch_size"]
        effective_neighbors_topk = min(train_params["neighbors_topk"], train_image_features.shape[0] - 1)
        checkpoint_loaded = True
    else:
        save_path = checkpoint_file(output_dir, checkpoint_name, create=True) if save_checkpoint else None
        predictions, training_history, effective_train_batch_size, effective_neighbors_topk = _train_sac_cluster_heads(
            train_image_features=train_image_features,
            train_text_features=retrieved_train,
            eval_image_features=eval_image_features,
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

    return SACOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=eval_labels.tolist() if eval_labels is not None else None,
        evaluation_split=eval_split,
        metadata={
            "variant": "sac_with_tac_text",
            "dataset": train_dataset.name,
            "source_dataset": train_dataset.name,
            "evaluation_dataset": eval_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": eval_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "text_construction": "tac_wordnet_noun_selection_and_retrieval",
            "original_sac_text_construction": "BLIP-2 captions encoded by SBERT; replaced for benchmark consistency.",
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
            "test_image_count": int(eval_image_features.shape[0]),
            "train_epochs": train_params["epochs"],
            "train_batch_size": effective_train_batch_size,
            "requested_train_batch_size": train_params["batch_size"],
            "neighbors_topk": effective_neighbors_topk,
            "requested_neighbors_topk": train_params["neighbors_topk"],
            "contrastive_temperature": train_params["temperature"],
            "save_checkpoint": save_checkpoint,
            "checkpoint_path": str(checkpoint_path) if (checkpoint_loaded or save_checkpoint) else None,
            "load_checkpoint": load_checkpoint_enabled(params, train_dataset.name),
            "checkpoint_loaded": checkpoint_loaded,
            "train_labels_available": train_labels is not None,
            "training_history": training_history,
        },
    )


def _retrieve_text_counterparts(
    image_features: np.ndarray,
    noun_embeddings: np.ndarray,
    temperature: float,
    batch_size: int,
    device,
    cache_path: Path,
    progress,
) -> np.ndarray:
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
        progress.step("Retrieving SAC text counterparts", batch_index, total_batches, noun="batch")
    retrieved_matrix = np.concatenate(retrieved, axis=0)
    np.save(cache_path, retrieved_matrix)
    progress.log(f"Saved retrieved text counterparts: {cache_path}")
    return retrieved_matrix


def _resolve_training_params(
    cluster_num: int,
    params: dict[str, Any],
    hyperparameter_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    high_cluster_count = cluster_num > 512
    defaults = {
        "epochs": 100 if high_cluster_count else 30,
        "batch_size": max(8192, cluster_num) if high_cluster_count else 512,
        "neighbors_topk": 50 if high_cluster_count else 20,
        "contrastive_temperature": 1.1,
    }
    if hyperparameter_defaults is not None:
        for key in ("epochs", "batch_size", "neighbors_topk", "contrastive_temperature"):
            if key in hyperparameter_defaults:
                defaults[key] = hyperparameter_defaults[key]
    if "entropy_coeff" in params:
        balance_weight = -float(params["entropy_coeff"])
    else:
        balance_weight = float(params.get("balance_weight", params.get("lambda_b", 4.0)))
    alignment_weight = float(params.get("alignment_weight", params.get("consist_coeff", params.get("lambda_a", 0.6))))
    beta = float(params.get("consistency_beta", params.get("beta", 0.5)))
    alpha = float(params.get("alpha", 0.9))

    if not 0.0 <= beta <= 1.0:
        raise ValueError("SAC consistency_beta/beta must be in [0, 1].")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SAC alpha must be in [0, 1].")

    return {
        "epochs": int(params.get("epochs", defaults["epochs"])),
        "batch_size": int(params.get("train_batch_size", params.get("batch_size", defaults["batch_size"]))),
        "neighbors_topk": int(params.get("neighbors_topk", params.get("topk", defaults["neighbors_topk"]))),
        "learning_rate": float(params.get("learning_rate", params.get("lr", 1.0e-3))),
        "temperature": float(
            params.get("contrastive_temperature", params.get("temperature", defaults["contrastive_temperature"]))
        ),
        "weight_scale": float(params.get("weight_scale", params.get("a", 1.0))),
        "alpha": alpha,
        "alignment_weight": alignment_weight,
        "balance_weight": balance_weight,
        "consistency_beta": beta,
        "weighted_contrastive": _as_bool(params.get("weighted_contrastive", True)),
        "seed": int(params.get("seed", 42)),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _train_sac_cluster_heads(
    train_image_features: np.ndarray,
    train_text_features: np.ndarray,
    eval_image_features: np.ndarray,
    cluster_num: int,
    epochs: int,
    batch_size: int,
    neighbors_topk: int,
    learning_rate: float,
    temperature: float,
    weight_scale: float,
    alpha: float,
    alignment_weight: float,
    balance_weight: float,
    consistency_beta: float,
    weighted_contrastive: bool,
    seed: int,
    device,
    output_dir: Path,
    image_neighbor_cache_key: str,
    text_neighbor_cache_key: str,
    checkpoint_path: Path | None = None,
    progress=None,
) -> tuple[np.ndarray, list[dict[str, float]], int, int]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = progress or get_progress_logger("sac")

    if train_image_features.shape[0] < 2:
        raise ValueError("SAC cluster-head training requires at least two training samples.")
    if train_image_features.shape[0] != train_text_features.shape[0]:
        raise ValueError("SAC requires train image and text counterpart matrices with the same number of rows.")
    if epochs <= 0:
        raise ValueError("SAC requires epochs > 0.")
    if batch_size <= 0:
        raise ValueError("SAC requires train_batch_size > 0.")
    if neighbors_topk <= 0:
        raise ValueError("SAC requires neighbors_topk/topk > 0.")

    effective_neighbors_topk = min(neighbors_topk, train_image_features.shape[0] - 1)
    if effective_neighbors_topk != neighbors_topk:
        progress.log(
            f"Reducing SAC neighbors_topk from {neighbors_topk} to {effective_neighbors_topk} "
            f"for {train_image_features.shape[0]} training sample(s)"
        )

    progress.log(f"Computing SAC nearest-neighbor pairs with topk={effective_neighbors_topk}")
    text_neighbors = _compute_neighbors(
        train_text_features,
        topk=effective_neighbors_topk,
        cache_path=cache_file_path(
            output_dir,
            f"{text_neighbor_cache_key}__text_neighbors_topk{effective_neighbors_topk}",
            ".npy",
        ),
        progress=progress,
        label="SAC text-neighbor search",
    )
    image_neighbors = _compute_neighbors(
        train_image_features,
        topk=effective_neighbors_topk,
        cache_path=cache_file_path(
            output_dir,
            f"{image_neighbor_cache_key}__image_neighbors_topk{effective_neighbors_topk}",
            ".npy",
        ),
        progress=progress,
        label="SAC image-neighbor search",
    )

    train_text_tensor = torch.from_numpy(train_text_features.astype("float32"))
    train_image_tensor = torch.from_numpy(train_image_features.astype("float32"))
    eval_image_tensor = torch.from_numpy(eval_image_features.astype("float32"))

    train_dataset = SACNeighborDataset(
        text_features=train_text_tensor,
        image_features=train_image_tensor,
        text_neighbor_indices=text_neighbors,
        image_neighbor_indices=image_neighbors,
        seed=seed,
    )
    eval_dataset = data_mod.TensorDataset(eval_image_tensor)

    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size != batch_size:
        progress.log(
            f"Reducing SAC train batch size from {batch_size} to {effective_batch_size} "
            f"for {len(train_dataset)} training sample(s)"
        )
    if effective_batch_size < cluster_num and balance_weight != 0.0:
        progress.log(
            f"SAC train batch size ({effective_batch_size}) is smaller than n_clusters ({cluster_num}); "
            "the batch-wise balance loss can destabilize high-class-count runs. "
            "Consider increasing train_batch_size or reducing balance_weight."
        )
    drop_last = len(train_dataset) > effective_batch_size and len(train_dataset) % effective_batch_size == 1
    train_loader = data_mod.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    eval_loader = data_mod.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = SACClusterHead(
        in_dim=train_image_features.shape[1],
        text_in_dim=train_text_features.shape[1],
        num_clusters=cluster_num,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    contrastive_loss = SACDataContrastiveLoss(temperature=temperature, weight_scale=weight_scale).to(device)

    history: list[dict[str, float]] = []
    efficiency_measurement = start_train_eval_measurement(device)
    progress.log(f"Training SAC cluster heads for {epochs} epoch(s)")
    for epoch in range(epochs):
        model.train()
        epoch_total = 0.0
        epoch_contrastive = 0.0
        epoch_intra = 0.0
        epoch_inter = 0.0
        epoch_alignment = 0.0
        epoch_balance = 0.0
        iterations = 0
        for (
            text_anchor,
            image_anchor,
            text_neighbor,
            image_neighbor,
            image_to_text,
            text_to_image,
        ) in train_loader:
            text_anchor = text_anchor.to(device)
            image_anchor = image_anchor.to(device)
            text_neighbor = text_neighbor.to(device)
            image_neighbor = image_neighbor.to(device)
            image_to_text = image_to_text.to(device)
            text_to_image = text_to_image.to(device)

            text_probs, image_probs = model(text_anchor, image_anchor)
            neighbor_text_probs, neighbor_image_probs = model(text_neighbor, image_neighbor)
            image_to_text_probs, text_to_image_probs = model(image_to_text, text_to_image)

            loss_intra = contrastive_loss(
                image_probs,
                text_to_image_probs,
                text_probs,
                image_to_text_probs,
                image_probs,
                text_to_image_probs,
                weighted=weighted_contrastive,
                alpha=alpha,
            )
            loss_inter_i2t = contrastive_loss(
                image_probs,
                neighbor_text_probs,
                text_probs,
                neighbor_image_probs,
                image_probs,
                neighbor_text_probs,
                weighted=weighted_contrastive,
                alpha=alpha,
            )
            loss_inter_t2i = contrastive_loss(
                text_probs,
                neighbor_image_probs,
                text_probs,
                neighbor_image_probs,
                image_probs,
                neighbor_text_probs,
                weighted=weighted_contrastive,
                alpha=alpha,
            )
            loss_contrastive = loss_intra + loss_inter_i2t + loss_inter_t2i
            loss_alignment = bidirectional_alignment_loss(text_probs, image_probs, consistency_beta)
            loss_balance = entropy_loss(text_probs) + entropy_loss(image_probs)
            loss = loss_contrastive + alignment_weight * loss_alignment - balance_weight * loss_balance

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_contrastive += float(loss_contrastive.item())
            epoch_intra += float(loss_intra.item())
            epoch_inter += float((loss_inter_i2t + loss_inter_t2i).item())
            epoch_alignment += float(loss_alignment.item())
            epoch_balance += float(loss_balance.item())
            iterations += 1

        epoch_record = {
            "epoch": float(epoch + 1),
            "loss_total": epoch_total / max(iterations, 1),
            "loss_contrastive": epoch_contrastive / max(iterations, 1),
            "loss_intra_contrastive": epoch_intra / max(iterations, 1),
            "loss_inter_contrastive": epoch_inter / max(iterations, 1),
            "loss_alignment": epoch_alignment / max(iterations, 1),
            "loss_balance": epoch_balance / max(iterations, 1),
        }
        history.append(epoch_record)
        progress.epoch(
            "SAC training",
            epoch + 1,
            epochs,
            metrics={
                "loss": epoch_record["loss_total"],
                "rc": epoch_record["loss_contrastive"],
                "align": epoch_record["loss_alignment"],
                "balance": epoch_record["loss_balance"],
            },
        )

    progress.log("Inferring SAC cluster assignments on the evaluation split")
    predictions = _infer_cluster_predictions(model, eval_loader, device)
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
    return predictions, history, effective_batch_size, effective_neighbors_topk


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


def _infer_sac_checkpoint(
    *,
    train_text_features: np.ndarray,
    eval_image_features: np.ndarray,
    cluster_num: int,
    batch_size: int,
    device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    model = SACClusterHead(
        in_dim=eval_image_features.shape[1],
        text_in_dim=train_text_features.shape[1],
        num_clusters=cluster_num,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    eval_tensor = torch.from_numpy(eval_image_features.astype("float32"))
    eval_loader = data_mod.DataLoader(data_mod.TensorDataset(eval_tensor), batch_size=batch_size, shuffle=False)
    predictions = _infer_cluster_predictions(model, eval_loader, device)
    history = payload.get("history", [])
    return predictions, history if isinstance(history, list) else []
