from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_file, load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.checkpointing import save_checkpoint_enabled
from vlm4cluster.methods.cluster_defaults import resolve_source_default_cluster_num
from vlm4cluster.methods.feature_cache import (
    cache_file_path,
    cache_key,
    image_dataset_cache_key,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
    wordnet_source_token,
)
from vlm4cluster.methods.magic.losses import MAGICDistillLoss, cluster_column_cross_entropy, entropy_loss
from vlm4cluster.methods.sac.core import (
    _resolve_sac_retrieval_temperature,
)
from vlm4cluster.methods.tac.core import (
    _compute_neighbors,
    _encode_image_dataset,
    _encode_wordnet_nouns,
    _normalize_dataset_name,
    _normalize_rows,
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


MAGIC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

MAGIC_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar100"): {"retrieval_temperature": 0.03, "batch_size": 2048},
    ("laion400m", "vitb32", "stl10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenet10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenetdogs"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "dtd"): {"retrieval_temperature": 0.03},
    ("laion400m", "vitb32", "ucf101"): {"retrieval_temperature": 0.01, "batch_size": 1024},
    ("laion400m", "vitb32", "places365standard"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb32", "imagenet"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb32", "aircraft"): {"retrieval_temperature": 0.05, "batch_size": 512},
    ("laion400m", "vitb32", "cars"): {"retrieval_temperature": 0.02, "batch_size": 1024},
    ("laion400m", "vitb32", "flowers"): {"retrieval_temperature": 0.001, "batch_size": 512},
    ("laion400m", "vitb32", "food"): {"retrieval_temperature": 0.03, "batch_size": 2048},
    ("laion400m", "vitb32", "pets"): {"retrieval_temperature": 0.008},
    ("laion400m", "vitb16", "cifar100"): {"retrieval_temperature": 0.03, "batch_size": 2048},
    ("laion400m", "vitb16", "stl10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "imagenet10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "imagenetdogs"): {"retrieval_temperature": 0.01, "batch_size": 1024},
    ("laion400m", "vitb16", "dtd"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitb16", "ucf101"): {"retrieval_temperature": 0.01, "batch_size": 1024},
    ("laion400m", "vitb16", "places365standard"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "imagenet"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitb16", "aircraft"): {"retrieval_temperature": 0.03, "batch_size": 512},
    ("laion400m", "vitb16", "cars"): {"retrieval_temperature": 0.01, "batch_size": 1024},
    ("laion400m", "vitb16", "flowers"): {"retrieval_temperature": 0.008, "batch_size": 512},
    ("laion400m", "vitb16", "food"): {"retrieval_temperature": 0.01, "batch_size": 4096},
    ("laion400m", "vitb16", "pets"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "cifar20"): {"retrieval_temperature": 0.04},
    ("laion400m", "vitl14", "cifar100"): {"batch_size": 2048},
    ("laion400m", "vitl14", "stl10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "imagenet10"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "imagenetdogs"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "dtd"): {"retrieval_temperature": 0.02},
    ("laion400m", "vitl14", "ucf101"): {"retrieval_temperature": 0.02, "batch_size": 1024},
    ("laion400m", "vitl14", "places365standard"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "imagenet"): {"retrieval_temperature": 0.01},
    ("laion400m", "vitl14", "aircraft"): {"batch_size": 512},
    ("laion400m", "vitl14", "cars"): {"retrieval_temperature": 0.02, "batch_size": 1024},
    ("laion400m", "vitl14", "flowers"): {"batch_size": 512},
    ("laion400m", "vitl14", "food"): {"retrieval_temperature": 0.01, "batch_size": 4096},
    ("laion400m", "vitl14", "pets"): {"retrieval_temperature": 0.03},
}


@dataclass(slots=True)
class MAGICOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


def _valid_num_heads(embed_dim: int, requested_heads: int) -> int:
    requested_heads = max(1, min(int(requested_heads), embed_dim))
    for heads in range(requested_heads, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


class MAGICClusterHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        num_clusters: int,
        *,
        bottleneck_ratio: float = 0.25,
        dropout_rate: float = 0.1,
        fusion_heads: int = 8,
    ) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_clusters = num_clusters
        self.text_projection = torch.nn.Linear(text_dim, image_dim) if text_dim != image_dim else torch.nn.Identity()
        heads = _valid_num_heads(image_dim, fusion_heads)
        self.cross_granularity_attention = torch.nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=heads,
            batch_first=True,
        )
        bottleneck_dim = max(1, int(round(image_dim * bottleneck_ratio)))
        self.image_adapter = torch.nn.Sequential(
            torch.nn.Linear(image_dim, bottleneck_dim),
            torch.nn.GELU(),
            torch.nn.Linear(bottleneck_dim, image_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.LayerNorm(image_dim),
        )
        self.text_adapter = torch.nn.Sequential(
            torch.nn.Linear(image_dim, bottleneck_dim),
            torch.nn.GELU(),
            torch.nn.Linear(bottleneck_dim, image_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.LayerNorm(image_dim),
        )
        self.image_head = torch.nn.Sequential(
            torch.nn.Linear(image_dim, image_dim),
            torch.nn.BatchNorm1d(image_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_dim, num_clusters),
            torch.nn.Softmax(dim=1),
        )
        self.text_head = torch.nn.Sequential(
            torch.nn.Linear(2 * image_dim, image_dim),
            torch.nn.BatchNorm1d(image_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_dim, num_clusters),
            torch.nn.Softmax(dim=1),
        )
        for module in (self.image_head[0], self.image_head[3], self.text_head[0], self.text_head[3]):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)

    def fuse_text(self, fine_text, coarse_text):
        torch = require_module("torch", "pip install torch")
        fine_text = self.text_projection(fine_text)
        coarse_text = self.text_projection(coarse_text)
        tokens = torch.stack((fine_text, coarse_text), dim=1)
        refined, _ = self.cross_granularity_attention(tokens, tokens, tokens)
        return torch.nn.functional.normalize(refined.mean(dim=1), dim=1)

    def forward(self, fine_text, coarse_text, image_features):
        torch = require_module("torch", "pip install torch")
        fused_text = self.fuse_text(fine_text, coarse_text)
        image_embedding = self.image_adapter(image_features)
        text_embedding = self.text_adapter(fused_text)
        image_probabilities = self.image_head(image_embedding)
        text_probabilities = self.text_head(torch.cat((text_embedding, image_embedding), dim=1))
        return text_probabilities, image_probabilities, text_embedding, image_embedding

    def predict_image(self, image_features):
        image_embedding = self.image_adapter(image_features)
        return self.image_head(image_embedding)


class MAGICNeighborDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        fine_text_features,
        coarse_text_features,
        image_features,
        text_neighbor_indices: np.ndarray,
        image_neighbor_indices: np.ndarray,
        seed: int | None = None,
    ) -> None:
        if fine_text_features.size(0) != coarse_text_features.size(0) or fine_text_features.size(0) != image_features.size(0):
            raise ValueError("MAGIC requires aligned fine-text, coarse-text, and image feature matrices.")
        self.fine_text_features = fine_text_features
        self.coarse_text_features = coarse_text_features
        self.image_features = image_features
        self.text_neighbor_indices = text_neighbor_indices
        self.image_neighbor_indices = image_neighbor_indices
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.image_features.size(0)

    def __getitem__(self, index: int):
        text_neighbor_index = int(self.rng.choice(self.text_neighbor_indices[index]))
        image_neighbor_index = int(self.rng.choice(self.image_neighbor_indices[index]))
        return (
            self.fine_text_features[index],
            self.coarse_text_features[index],
            self.image_features[index],
            self.fine_text_features[text_neighbor_index],
            self.coarse_text_features[text_neighbor_index],
            self.image_features[text_neighbor_index],
            self.fine_text_features[image_neighbor_index],
            self.coarse_text_features[image_neighbor_index],
            self.image_features[image_neighbor_index],
        )


def _resolve_magic_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_magic_profile_token(openclip_pretraining),
        _normalize_magic_profile_token(openclip_backbone),
        _normalize_magic_profile_dataset(dataset_name),
    )
    return dict(MAGIC_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_magic_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    profile_key = (
        _normalize_magic_profile_token(openclip_pretraining),
        _normalize_magic_profile_token(openclip_backbone),
        _normalize_magic_profile_dataset(dataset_name),
    )
    if profile_key not in MAGIC_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_magic_retrieval_temperature(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    return _resolve_sac_retrieval_temperature(params, defaults)


def _normalize_magic_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_magic_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in MAGIC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _retrieve_magic_text_granularities(
    image_features: np.ndarray,
    noun_embeddings: np.ndarray,
    *,
    retrieval_temperature: float,
    coarse_topk: int,
    coarse_weight_mode: str,
    batch_size: int,
    device,
    cache_path: Path,
    progress,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        progress.log(f"Loading cached MAGIC text granularities: {cache_path}")
        payload = np.load(cache_path)
        return payload["fine"], payload["coarse"], payload["top_indices"]

    if coarse_topk <= 0:
        raise ValueError("MAGIC coarse_topk/num_coarse_nouns must be > 0.")
    if noun_embeddings.shape[0] == 0:
        raise ValueError("MAGIC text construction requires at least one selected noun embedding.")
    effective_topk = min(coarse_topk, noun_embeddings.shape[0])
    torch = require_module("torch", "pip install torch")
    noun_tensor = torch.from_numpy(noun_embeddings.astype("float32")).to(device)
    noun_tensor = torch.nn.functional.normalize(noun_tensor, dim=1)
    image_tensor = torch.from_numpy(image_features.astype("float32")).to(device)
    image_tensor = torch.nn.functional.normalize(image_tensor, dim=1)

    fine_batches = []
    coarse_batches = []
    top_index_batches = []
    total_batches = max(1, (image_tensor.size(0) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, image_tensor.size(0), batch_size), start=1):
        batch = image_tensor[start : start + batch_size]
        similarity = batch @ noun_tensor.t()

        fine_weights = torch.softmax(similarity / retrieval_temperature, dim=1)
        fine = torch.nn.functional.normalize(fine_weights @ noun_tensor, dim=1)

        top_values, top_indices = torch.topk(similarity, k=effective_topk, dim=1)
        top_nouns = noun_tensor[top_indices]
        if coarse_weight_mode == "softmax":
            coarse_weights = torch.softmax(top_values / retrieval_temperature, dim=1)
        elif coarse_weight_mode == "similarity":
            positive_values = torch.clamp(top_values, min=0.0)
            denominator = positive_values.sum(dim=1, keepdim=True)
            fallback_weights = torch.softmax(top_values / retrieval_temperature, dim=1)
            coarse_weights = torch.where(
                denominator > 1e-12,
                positive_values / torch.clamp(denominator, min=1e-12),
                fallback_weights,
            )
        else:
            raise ValueError("MAGIC coarse_weight_mode must be 'similarity' or 'softmax'.")
        coarse = torch.sum(top_nouns * coarse_weights.unsqueeze(-1), dim=1)
        coarse = torch.nn.functional.normalize(coarse, dim=1)

        fine_batches.append(fine.cpu().numpy().astype("float32"))
        coarse_batches.append(coarse.cpu().numpy().astype("float32"))
        top_index_batches.append(top_indices.cpu().numpy().astype("int64"))
        progress.step("Constructing MAGIC text granularities", batch_index, total_batches, noun="batch")

    fine_matrix = np.concatenate(fine_batches, axis=0)
    coarse_matrix = np.concatenate(coarse_batches, axis=0)
    top_indices_matrix = np.concatenate(top_index_batches, axis=0)
    np.savez_compressed(cache_path, fine=fine_matrix, coarse=coarse_matrix, top_indices=top_indices_matrix)
    progress.log(f"Saved MAGIC text granularities: {cache_path}")
    return fine_matrix, coarse_matrix, top_indices_matrix


def run_magic_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> MAGICOutputs:
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger("magic", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "magic",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="MAGIC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if train_dataset.num_classes == 0:
        raise ValueError("MAGIC requires datasets with known class names on the source train split.")

    cluster_num, cluster_num_source = resolve_source_default_cluster_num(
        params,
        train_dataset,
        method_name="MAGIC",
    )
    if cluster_num <= 0:
        raise ValueError("MAGIC requires n_clusters > 0.")

    hyperparameter_defaults = _resolve_magic_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    hyperparameter_profile_source = _resolve_magic_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    image_batch_size = int(params.get("image_batch_size", 256))
    text_batch_size = int(params.get("text_batch_size", 2048))
    retrieval_batch_size = int(params.get("retrieval_batch_size", 8192))
    selected_nouns_per_center = int(params.get("selected_nouns_per_center", 5))
    retrieval_temperature = _resolve_magic_retrieval_temperature(params, hyperparameter_defaults)
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
        raise ValueError("MAGIC TAC-style noun filtering selected zero nouns.")

    coarse_topk = int(params.get("coarse_topk", params.get("num_coarse_nouns", params.get("coarse_k", 3))))
    coarse_weight_mode = str(params.get("coarse_weight_mode", "similarity")).strip().lower()
    prediction_head = _resolve_prediction_head(params)
    text_cache_key = cache_key(
        selection_cache_key,
        f"retrieval{retrieval_temperature}",
        f"coarse{coarse_topk}",
        f"weight{coarse_weight_mode}",
    )
    with progress.stage("Constructing MAGIC train text granularities from TAC-selected nouns"):
        train_text_fine, train_text_coarse, train_top_indices = _retrieve_magic_text_granularities(
            image_features=train_image_features,
            noun_embeddings=selected_noun_embeddings,
            retrieval_temperature=retrieval_temperature,
            coarse_topk=coarse_topk,
            coarse_weight_mode=coarse_weight_mode,
            batch_size=retrieval_batch_size,
            device=device,
            cache_path=cache_file_path(output_dir, f"{text_cache_key}__train_magic_text", ".npz"),
            progress=progress,
        )
    eval_text_fine = None
    eval_text_coarse = None
    if bool(params.get("cache_eval_text", True)) or prediction_head in {"text", "both"}:
        eval_text_cache_key = cache_key(image_dataset_cache_key(eval_dataset), "source", text_cache_key)
        with progress.stage("Constructing MAGIC evaluation text granularities from TAC-selected nouns"):
            eval_text_fine, eval_text_coarse, _ = _retrieve_magic_text_granularities(
                image_features=eval_image_features,
                noun_embeddings=selected_noun_embeddings,
                retrieval_temperature=retrieval_temperature,
                coarse_topk=coarse_topk,
                coarse_weight_mode=coarse_weight_mode,
                batch_size=retrieval_batch_size,
                device=device,
                cache_path=cache_file_path(output_dir, f"{eval_text_cache_key}__eval_magic_text", ".npz"),
                progress=progress,
            )

    train_params = _resolve_training_params(
        cluster_num=cluster_num,
        params=params,
        hyperparameter_defaults=hyperparameter_defaults,
    )
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)
    checkpoint_name = f"{cache_key(train_feature_cache_key, text_cache_key, f'k{cluster_num}')}__magic_final.pt"
    checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    checkpoint_loaded = False
    if load_checkpoint_enabled(params, train_dataset.name) and checkpoint_path.exists():
        progress.log(f"Loading MAGIC checkpoint and skipping head training: {checkpoint_path}")
        predictions, extra_predictions, training_history = _infer_magic_checkpoint(
            eval_image_features=eval_image_features,
            eval_text_fine=eval_text_fine,
            eval_text_coarse=eval_text_coarse,
            batch_size=train_params["batch_size"],
            device=device,
            checkpoint_path=checkpoint_path,
            prediction_head=prediction_head,
        )
        effective_train_batch_size = train_params["batch_size"]
        effective_neighbors_topk = min(train_params["neighbors_topk"], train_image_features.shape[0] - 1)
        checkpoint_loaded = True
    else:
        save_path = checkpoint_file(output_dir, checkpoint_name, create=True) if save_checkpoint else None
        (
            predictions,
            extra_predictions,
            training_history,
            effective_train_batch_size,
            effective_neighbors_topk,
        ) = _train_magic_cluster_heads(
            train_image_features=train_image_features,
            train_text_fine=train_text_fine,
            train_text_coarse=train_text_coarse,
            eval_image_features=eval_image_features,
            eval_text_fine=eval_text_fine,
            eval_text_coarse=eval_text_coarse,
            cluster_num=cluster_num,
            device=device,
            output_dir=output_dir,
            image_neighbor_cache_key=train_feature_cache_key,
            text_neighbor_cache_key=text_cache_key,
            checkpoint_path=save_path,
            progress=progress,
            prediction_head=prediction_head,
            **train_params,
        )
        checkpoint_path = save_path or checkpoint_path

    top_noun_examples = []
    for row in train_top_indices[:5]:
        top_noun_examples.append([candidate_nouns[selected_noun_indices[int(index)]] for index in row])

    return MAGICOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=eval_labels.tolist() if eval_labels is not None else None,
        evaluation_split=eval_split,
        metadata={
            "variant": "magic_with_tac_text",
            "dataset": train_dataset.name,
            "source_dataset": train_dataset.name,
            "evaluation_dataset": eval_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": eval_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "text_construction": "tac_wordnet_noun_selection_with_magic_dense_and_topk_granularities",
            "original_magic_text_construction": "Qwen2.5-VL generated captions plus spaCy concepts; replaced for benchmark consistency.",
            "wordnet_csv": str(wordnet_csv),
            "cluster_num": cluster_num,
            "cluster_num_source": cluster_num_source,
            "retrieval_temperature": retrieval_temperature,
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "semantic_clusters": effective_semantic_clusters,
            "semantic_clusters_source": _resolve_semantic_cluster_source(train_dataset.name, semantic_clusters),
            "selected_noun_count": int(selected_noun_embeddings.shape[0]),
            "selected_noun_examples": [candidate_nouns[index] for index in selected_noun_indices[:10]],
            "coarse_topk": coarse_topk,
            "coarse_weight_mode": coarse_weight_mode,
            "prediction_head": prediction_head,
            "primary_prediction_head": "text" if prediction_head == "text" else "image",
            "extra_predictions": extra_predictions,
            "train_top_noun_examples": top_noun_examples,
            "train_image_count": int(train_image_features.shape[0]),
            "test_image_count": int(eval_image_features.shape[0]),
            "train_epochs": train_params["epochs"],
            "train_batch_size": effective_train_batch_size,
            "requested_train_batch_size": train_params["batch_size"],
            "neighbors_topk": effective_neighbors_topk,
            "requested_neighbors_topk": train_params["neighbors_topk"],
            "distill_temperature": train_params["distill_temperature"],
            "alignment_weight": train_params["alignment_weight"],
            "balance_weight": train_params["balance_weight"],
            "bottleneck_ratio": train_params["bottleneck_ratio"],
            "save_checkpoint": save_checkpoint,
            "checkpoint_path": str(checkpoint_path) if (checkpoint_loaded or save_checkpoint) else None,
            "load_checkpoint": load_checkpoint_enabled(params, train_dataset.name),
            "checkpoint_loaded": checkpoint_loaded,
            "train_labels_available": train_labels is not None,
            "training_history": training_history,
        },
    )


def _resolve_prediction_head(params: dict[str, Any]) -> str:
    prediction_head = str(params.get("prediction_head", params.get("head", "both"))).strip().lower()
    aliases = {
        "i": "image",
        "img": "image",
        "image_head": "image",
        "t": "text",
        "txt": "text",
        "text_head": "text",
        "all": "both",
    }
    prediction_head = aliases.get(prediction_head, prediction_head)
    if prediction_head not in {"image", "text", "both"}:
        raise ValueError("MAGIC prediction_head/head must be one of: image, text, both.")
    return prediction_head


def _resolve_training_params(
    cluster_num: int,
    params: dict[str, Any],
    hyperparameter_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    high_cluster_count = cluster_num >= 100
    defaults: dict[str, Any] = {
        "epochs": 100,
        "batch_size": 8192 if high_cluster_count else 512,
        "neighbors_topk": 50,
        "distill_temperature": 1.1,
        "learning_rate": 1.0e-4,
        "bottleneck_ratio": 0.25,
        "dropout_rate": 0.1,
        "fusion_heads": 8,
    }
    if hyperparameter_defaults is not None:
        for key in ("epochs", "batch_size", "neighbors_topk"):
            if key in hyperparameter_defaults:
                defaults[key] = hyperparameter_defaults[key]
    if "entropy_coeff" in params:
        balance_weight = -float(params["entropy_coeff"])
    else:
        balance_weight = float(params.get("balance_weight", params.get("mu", 5.0)))
    alignment_weight = float(params.get("alignment_weight", params.get("consist_coeff", params.get("lambda", 0.6))))
    bottleneck_ratio = float(params.get("bottleneck_ratio", params.get("r", params.get("l", defaults["bottleneck_ratio"]))))
    if not 0.0 < bottleneck_ratio <= 1.0:
        raise ValueError("MAGIC bottleneck_ratio/r/l must be in (0, 1].")
    return {
        "epochs": int(params.get("epochs", defaults["epochs"])),
        "batch_size": int(params.get("train_batch_size", params.get("batch_size", defaults["batch_size"]))),
        "neighbors_topk": int(params.get("neighbors_topk", params.get("topk", defaults["neighbors_topk"]))),
        "learning_rate": float(params.get("learning_rate", params.get("lr", defaults["learning_rate"]))),
        "distill_temperature": float(
            params.get("distill_temperature", params.get("temperature", defaults["distill_temperature"]))
        ),
        "alignment_weight": alignment_weight,
        "balance_weight": balance_weight,
        "bottleneck_ratio": bottleneck_ratio,
        "dropout_rate": float(params.get("dropout_rate", defaults["dropout_rate"])),
        "fusion_heads": int(params.get("fusion_heads", defaults["fusion_heads"])),
        "seed": int(params.get("seed", 42)),
    }


def _train_magic_cluster_heads(
    train_image_features: np.ndarray,
    train_text_fine: np.ndarray,
    train_text_coarse: np.ndarray,
    eval_image_features: np.ndarray,
    eval_text_fine: np.ndarray | None,
    eval_text_coarse: np.ndarray | None,
    cluster_num: int,
    epochs: int,
    batch_size: int,
    neighbors_topk: int,
    learning_rate: float,
    distill_temperature: float,
    alignment_weight: float,
    balance_weight: float,
    bottleneck_ratio: float,
    dropout_rate: float,
    fusion_heads: int,
    seed: int,
    device,
    output_dir: Path,
    image_neighbor_cache_key: str,
    text_neighbor_cache_key: str,
    checkpoint_path: Path | None = None,
    progress=None,
    prediction_head: str = "both",
) -> tuple[np.ndarray, dict[str, list[int]], list[dict[str, float]], int, int]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = progress or get_progress_logger("magic")

    if train_image_features.shape[0] < 2:
        raise ValueError("MAGIC cluster-head training requires at least two training samples.")
    if train_image_features.shape[0] != train_text_fine.shape[0] or train_image_features.shape[0] != train_text_coarse.shape[0]:
        raise ValueError("MAGIC requires aligned train image/fine-text/coarse-text matrices.")
    if epochs <= 0:
        raise ValueError("MAGIC requires epochs > 0.")
    if batch_size <= 0:
        raise ValueError("MAGIC requires train_batch_size > 0.")
    if neighbors_topk <= 0:
        raise ValueError("MAGIC requires neighbors_topk/topk > 0.")

    effective_neighbors_topk = min(neighbors_topk, train_image_features.shape[0] - 1)
    if effective_neighbors_topk != neighbors_topk:
        progress.log(
            f"Reducing MAGIC neighbors_topk from {neighbors_topk} to {effective_neighbors_topk} "
            f"for {train_image_features.shape[0]} training sample(s)"
        )

    initial_text_neighbors_features = _normalize_rows((train_text_fine + train_text_coarse).astype("float32"))
    progress.log(f"Computing MAGIC nearest-neighbor pairs with topk={effective_neighbors_topk}")
    text_neighbors = _compute_neighbors(
        initial_text_neighbors_features,
        topk=effective_neighbors_topk,
        cache_path=cache_file_path(
            output_dir,
            f"{text_neighbor_cache_key}__text_neighbors_topk{effective_neighbors_topk}",
            ".npy",
        ),
        progress=progress,
        label="MAGIC text-neighbor search",
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
        label="MAGIC image-neighbor search",
    )

    train_fine_tensor = torch.from_numpy(train_text_fine.astype("float32"))
    train_coarse_tensor = torch.from_numpy(train_text_coarse.astype("float32"))
    train_image_tensor = torch.from_numpy(train_image_features.astype("float32"))
    eval_image_tensor = torch.from_numpy(eval_image_features.astype("float32"))
    train_dataset = MAGICNeighborDataset(
        fine_text_features=train_fine_tensor,
        coarse_text_features=train_coarse_tensor,
        image_features=train_image_tensor,
        text_neighbor_indices=text_neighbors,
        image_neighbor_indices=image_neighbors,
        seed=seed,
    )
    eval_loader = _build_eval_loader(
        data_mod=data_mod,
        eval_image_tensor=eval_image_tensor,
        eval_text_fine=eval_text_fine,
        eval_text_coarse=eval_text_coarse,
        batch_size=batch_size,
        prediction_head=prediction_head,
    )

    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size != batch_size:
        progress.log(
            f"Reducing MAGIC train batch size from {batch_size} to {effective_batch_size} "
            f"for {len(train_dataset)} training sample(s)"
        )
    if effective_batch_size < cluster_num and balance_weight != 0.0:
        progress.log(
            f"MAGIC train batch size ({effective_batch_size}) is smaller than n_clusters ({cluster_num}); "
            "the batch-wise balance loss can be noisy. Consider increasing train_batch_size or reducing balance_weight."
        )
    drop_last = len(train_dataset) > effective_batch_size and len(train_dataset) % effective_batch_size == 1
    train_loader = data_mod.DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=drop_last)
    model = MAGICClusterHead(
        image_dim=train_image_features.shape[1],
        text_dim=train_text_fine.shape[1],
        num_clusters=cluster_num,
        bottleneck_ratio=bottleneck_ratio,
        dropout_rate=dropout_rate,
        fusion_heads=fusion_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    distill_loss = MAGICDistillLoss(class_num=cluster_num, temperature=distill_temperature).to(device)

    history: list[dict[str, float]] = []
    efficiency_measurement = start_train_eval_measurement(device)
    progress.log(f"Training MAGIC cluster heads for {epochs} epoch(s)")
    for epoch in range(epochs):
        model.train()
        epoch_total = 0.0
        epoch_distill = 0.0
        epoch_intra = 0.0
        epoch_inter = 0.0
        epoch_alignment = 0.0
        epoch_balance = 0.0
        iterations = 0
        for (
            fine_anchor,
            coarse_anchor,
            image_anchor,
            fine_text_neighbor,
            coarse_text_neighbor,
            image_text_neighbor,
            fine_image_neighbor,
            coarse_image_neighbor,
            image_image_neighbor,
        ) in train_loader:
            fine_anchor = fine_anchor.to(device)
            coarse_anchor = coarse_anchor.to(device)
            image_anchor = image_anchor.to(device)
            fine_text_neighbor = fine_text_neighbor.to(device)
            coarse_text_neighbor = coarse_text_neighbor.to(device)
            image_text_neighbor = image_text_neighbor.to(device)
            fine_image_neighbor = fine_image_neighbor.to(device)
            coarse_image_neighbor = coarse_image_neighbor.to(device)
            image_image_neighbor = image_image_neighbor.to(device)

            text_probs, image_probs, _, _ = model(fine_anchor, coarse_anchor, image_anchor)
            text_neighbor_probs, image_text_neighbor_probs, _, _ = model(
                fine_text_neighbor,
                coarse_text_neighbor,
                image_text_neighbor,
            )
            image_neighbor_text_probs, image_neighbor_probs, _, _ = model(
                fine_image_neighbor,
                coarse_image_neighbor,
                image_image_neighbor,
            )

            loss_intra = distill_loss(image_probs, image_text_neighbor_probs) + distill_loss(
                text_probs,
                image_neighbor_text_probs,
            )
            loss_inter = distill_loss(image_probs, text_neighbor_probs) + distill_loss(text_probs, image_neighbor_probs)
            loss_distill = loss_intra + loss_inter
            loss_alignment = cluster_column_cross_entropy(image_probs, text_probs)
            loss_balance = entropy_loss(image_probs) + entropy_loss(text_probs)
            loss = loss_distill + alignment_weight * loss_alignment - balance_weight * loss_balance

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_distill += float(loss_distill.item())
            epoch_intra += float(loss_intra.item())
            epoch_inter += float(loss_inter.item())
            epoch_alignment += float(loss_alignment.item())
            epoch_balance += float(loss_balance.item())
            iterations += 1

        epoch_record = {
            "epoch": float(epoch + 1),
            "loss_total": epoch_total / max(iterations, 1),
            "loss_distill": epoch_distill / max(iterations, 1),
            "loss_intra_distill": epoch_intra / max(iterations, 1),
            "loss_inter_distill": epoch_inter / max(iterations, 1),
            "loss_alignment": epoch_alignment / max(iterations, 1),
            "loss_balance": epoch_balance / max(iterations, 1),
        }
        history.append(epoch_record)
        progress.epoch(
            "MAGIC training",
            epoch + 1,
            epochs,
            metrics={
                "loss": epoch_record["loss_total"],
                "distill": epoch_record["loss_distill"],
                "align": epoch_record["loss_alignment"],
                "balance": epoch_record["loss_balance"],
            },
        )

    progress.log(f"Inferring MAGIC {prediction_head} cluster assignments on the evaluation split")
    predictions_by_head = _infer_predictions_by_head(model, eval_loader, device, prediction_head)
    efficiency_measurement.stop()
    primary_head = "text" if prediction_head == "text" else "image"
    predictions = predictions_by_head[primary_head]
    extra_predictions = {
        head: values.tolist()
        for head, values in predictions_by_head.items()
        if head != primary_head
    }
    if checkpoint_path is not None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cluster_num": cluster_num,
                "image_dim": train_image_features.shape[1],
                "text_dim": train_text_fine.shape[1],
                "bottleneck_ratio": bottleneck_ratio,
                "dropout_rate": dropout_rate,
                "fusion_heads": fusion_heads,
                "epochs": epochs,
                "history": history,
            },
            checkpoint_path,
        )
    return predictions, extra_predictions, history, effective_batch_size, effective_neighbors_topk


def _infer_magic_checkpoint(
    *,
    eval_image_features: np.ndarray,
    eval_text_fine: np.ndarray | None,
    eval_text_coarse: np.ndarray | None,
    batch_size: int,
    device,
    checkpoint_path: Path,
    prediction_head: str,
) -> tuple[np.ndarray, dict[str, list[int]], list[dict[str, float]]]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    model = MAGICClusterHead(
        image_dim=int(payload["image_dim"]),
        text_dim=int(payload["text_dim"]),
        num_clusters=int(payload["cluster_num"]),
        bottleneck_ratio=float(payload.get("bottleneck_ratio", 0.25)),
        dropout_rate=float(payload.get("dropout_rate", 0.1)),
        fusion_heads=int(payload.get("fusion_heads", 8)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    eval_tensor = torch.from_numpy(eval_image_features.astype("float32"))
    eval_loader = _build_eval_loader(
        data_mod=data_mod,
        eval_image_tensor=eval_tensor,
        eval_text_fine=eval_text_fine,
        eval_text_coarse=eval_text_coarse,
        batch_size=batch_size,
        prediction_head=prediction_head,
    )
    predictions_by_head = _infer_predictions_by_head(model, eval_loader, device, prediction_head)
    primary_head = "text" if prediction_head == "text" else "image"
    predictions = predictions_by_head[primary_head]
    extra_predictions = {
        head: values.tolist()
        for head, values in predictions_by_head.items()
        if head != primary_head
    }
    history = payload.get("history", [])
    return predictions, extra_predictions, history if isinstance(history, list) else []


def _build_eval_loader(
    *,
    data_mod,
    eval_image_tensor,
    eval_text_fine: np.ndarray | None,
    eval_text_coarse: np.ndarray | None,
    batch_size: int,
    prediction_head: str,
):
    if prediction_head in {"text", "both"}:
        if eval_text_fine is None or eval_text_coarse is None:
            raise ValueError("MAGIC prediction_head=text/both requires evaluation text granularities.")
        torch = require_module("torch", "pip install torch")
        fine_tensor = torch.from_numpy(eval_text_fine.astype("float32"))
        coarse_tensor = torch.from_numpy(eval_text_coarse.astype("float32"))
        eval_dataset = data_mod.TensorDataset(fine_tensor, coarse_tensor, eval_image_tensor)
    else:
        eval_dataset = data_mod.TensorDataset(eval_image_tensor)
    return data_mod.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def _infer_predictions_by_head(model, dataloader, device, prediction_head: str) -> dict[str, np.ndarray]:
    torch = require_module("torch", "pip install torch")
    model.eval()
    image_predictions = []
    text_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            if prediction_head in {"text", "both"}:
                fine_text, coarse_text, image_features = batch
                fine_text = fine_text.to(device)
                coarse_text = coarse_text.to(device)
                image_features = image_features.to(device)
                text_probabilities, image_probabilities, _, _ = model(fine_text, coarse_text, image_features)
                text_predictions.append(torch.argmax(text_probabilities, dim=1).cpu().numpy())
            else:
                (image_features,) = batch
                image_features = image_features.to(device)
                image_probabilities = model.predict_image(image_features)
            image_predictions.append(torch.argmax(image_probabilities, dim=1).cpu().numpy())

    predictions = {"image": np.concatenate(image_predictions, axis=0)}
    if text_predictions:
        predictions["text"] = np.concatenate(text_predictions, axis=0)
    return predictions
