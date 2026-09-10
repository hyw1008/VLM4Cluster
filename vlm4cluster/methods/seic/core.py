from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label
from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_file, load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.checkpointing import save_checkpoint_enabled
from vlm4cluster.methods.cluster_defaults import resolve_source_default_cluster_num
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
from vlm4cluster.methods.seic.lora import inject_lora_into_visual_attention
from vlm4cluster.methods.seic.losses import alignment_loss, self_enhancement_loss
from vlm4cluster.methods.tac.core import (
    _load_tac_wordnet_nouns,
    _normalize_dataset_name,
    _resolve_domain_shift_datasets,
    _resolve_tac_wordnet_csv,
)
from vlm4cluster.models import OpenCLIPBundle, load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import FAISS_KMEANS_CACHE_TOKEN, run_faiss_kmeans, search_topk
from vlm4cluster.utils.progress import get_progress_logger


SEIC_WORDNET_PROMPT_TEMPLATE_KEY = COMMON_WORDNET_PROMPT_TEMPLATE_KEY
SEIC_WORDNET_PROMPT_BUILDERS = COMMON_WORDNET_PROMPT_BUILDERS

SEIC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

SEIC_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "dtd"): {
        "text_weight_temperature": 0.008,
        "assignment_temperature": 0.2,
    },
    ("laion400m", "vitb32", "ucf101"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitb32", "aircraft"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitb32", "cars"): {"text_weight_temperature": 0.05},
    ("laion400m", "vitb32", "pets"): {
        "text_weight_temperature": 0.008,
        "center_temperature": 0.4,
    },
    ("laion400m", "vitb16", "cifar10"): {"text_weight_temperature": 0.04},
    ("laion400m", "vitb16", "dtd"): {
        "text_weight_temperature": 0.008,
        "assignment_temperature": 0.2,
    },
    ("laion400m", "vitb16", "ucf101"): {"text_weight_temperature": 0.03},
    ("laion400m", "vitb16", "aircraft"): {
        "text_weight_temperature": 0.04,
        "center_temperature": 0.6,
    },
    ("laion400m", "vitb16", "cars"): {
        "text_weight_temperature": 0.05,
        "assignment_temperature": 0.4,
    },
    ("laion400m", "vitb16", "food"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitb16", "flowers"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitl14", "cifar20"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitl14", "stl10"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitl14", "dtd"): {
        "text_weight_temperature": 0.008,
        "assignment_temperature": 0.2,
    },
    ("laion400m", "vitl14", "ucf101"): {"text_weight_temperature": 0.02},
    ("laion400m", "vitl14", "aircraft"): {"text_weight_temperature": 0.03},
    ("laion400m", "vitl14", "cars"): {"text_weight_temperature": 0.04},
    ("laion400m", "vitl14", "food"): {"text_weight_temperature": 0.04},
}


@dataclass(slots=True)
class SEICOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class SEICHeads(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_dim: int, num_clusters: int, *, instance_temperature: float = 0.07) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_clusters = int(num_clusters)
        self.image_projection = torch.nn.Linear(in_dim, in_dim)
        self.text_projection = torch.nn.Linear(in_dim, in_dim)
        self.image_clustering = torch.nn.Linear(in_dim, num_clusters)
        self.text_clustering = torch.nn.Linear(in_dim, num_clusters)
        self.instance_logit_scale = torch.nn.Parameter(
            torch.ones((), dtype=torch.float32) * np.log(1.0 / float(instance_temperature))
        )
        self._init_weights()

    def _init_weights(self) -> None:
        torch = require_module("torch", "pip install torch")
        for module in (self.image_projection, self.text_projection, self.image_clustering, self.text_clustering):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward_image_features(self, features):
        torch = require_module("torch", "pip install torch")
        projected = torch.nn.functional.normalize(self.image_projection(features.float()), dim=1)
        logits = self.image_clustering(projected)
        probabilities = torch.softmax(logits, dim=1)
        return projected, logits, probabilities

    def forward_text_features(self, features):
        torch = require_module("torch", "pip install torch")
        projected = torch.nn.functional.normalize(self.text_projection(features.float()), dim=1)
        logits = self.text_clustering(projected)
        probabilities = torch.softmax(logits, dim=1)
        return projected, logits, probabilities

    def forward_pair(self, image_features, text_features):
        image_projected, image_logits, image_probabilities = self.forward_image_features(image_features)
        text_projected, text_logits, text_probabilities = self.forward_text_features(text_features)
        return {
            "image_projected": image_projected,
            "image_logits": image_logits,
            "image_probabilities": image_probabilities,
            "text_projected": text_projected,
            "text_logits": text_logits,
            "text_probabilities": text_probabilities,
        }

    def set_stage2_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.image_projection, self.image_clustering):
            for parameter in module.parameters():
                parameter.requires_grad = True


class SEICVisionModel(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, clip_model, heads: SEICHeads) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.clip_model = clip_model
        self.heads = heads

    def encode_image(self, images):
        torch = require_module("torch", "pip install torch")
        features = self.clip_model.encode_image(images).float()
        return torch.nn.functional.normalize(features, dim=1)

    def forward(self, images):
        features = self.encode_image(images)
        projected, logits, probabilities = self.heads.forward_image_features(features)
        return projected, logits, probabilities

    def train(self, mode: bool = True):
        super().train(mode)
        return self


class SEICSelfEnhancementDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, dataset: LoadedImageDataset, weak_transform, strong_transform) -> None:
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        transforms = require_module("torchvision.transforms", "pip install torch torchvision")
        self.to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.dataset.sample_indices)

    def __getitem__(self, index: int):
        dataset_index = self.dataset.sample_indices[index]
        image, raw_label = self.dataset.dataset[dataset_index]
        if hasattr(image, "detach"):
            image = self.to_pil(image)
        return {
            "weak": self.weak_transform(image),
            "strong": self.strong_transform(image),
            "target": get_image_label(self.dataset, dataset_index, raw_label),
        }


class SEICEvalDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, dataset: LoadedImageDataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform
        transforms = require_module("torchvision.transforms", "pip install torch torchvision")
        self.to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.dataset.sample_indices)

    def __getitem__(self, index: int):
        dataset_index = self.dataset.sample_indices[index]
        image, raw_label = self.dataset.dataset[dataset_index]
        if hasattr(image, "detach"):
            image = self.to_pil(image)
        return {
            "image": self.transform(image),
            "target": get_image_label(self.dataset, dataset_index, raw_label),
        }


def run_seic_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> SEICOutputs:
    torch = require_module("torch", "pip install torch")
    progress = get_progress_logger("seic", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "seic",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="SEIC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if train_dataset.num_classes == 0:
        raise ValueError("SEIC requires source-train datasets with known class names.")
    if eval_dataset.labels is None:
        raise ValueError("SEIC requires labels on the evaluation split.")

    cluster_num, cluster_num_source = resolve_source_default_cluster_num(
        params,
        train_dataset,
        method_name="SEIC",
    )
    if cluster_num <= 0:
        raise ValueError("SEIC requires n_clusters > 0.")
    if cluster_num > train_dataset.num_samples:
        raise ValueError(
            f"SEIC n_clusters ({cluster_num}) cannot exceed source-train sample count ({train_dataset.num_samples})."
        )

    resolved = _resolve_seic_params(
        params,
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    hyperparameter_profile_source = _resolve_seic_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    random_state = params.get("random_state", params.get("seed"))
    stage2_enabled = bool(resolved["stage2"])
    wordnet_csv = _resolve_tac_wordnet_csv(params)
    text_cache_key = cache_key(
        image_dataset_cache_key(train_dataset),
        bundle.spec.cache_key,
        wordnet_source_token(wordnet_csv),
        SEIC_WORDNET_PROMPT_TEMPLATE_KEY,
        FAISS_KMEANS_CACHE_TOKEN,
        f"k{cluster_num}",
        f"k1{resolved['k1']}",
        f"k2{resolved['k2']}",
        f"tw{resolved['text_weight_temperature']}",
        f"seed{random_state}",
    )

    with progress.stage(f"Encoding evaluation images '{eval_dataset.name}/{eval_split}'"):
        eval_image_features, eval_labels = _encode_normalized_images(
            eval_dataset,
            bundle=bundle,
            batch_size=resolved["image_batch_size"],
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )
    heads = SEICHeads(
        in_dim=int(eval_image_features.shape[1]),
        num_clusters=cluster_num,
        instance_temperature=resolved["instance_temperature"],
    ).to(device)

    checkpoint_name = _seic_stage1_checkpoint_name(
        text_cache_key=text_cache_key,
        in_dim=int(eval_image_features.shape[1]),
        params=resolved,
    )
    stage1_checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    stage1_save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)
    stage1_load_checkpoint = load_checkpoint_enabled(params, train_dataset.name)
    stage1_checkpoint_loaded = False
    stage1_history: list[dict[str, float]] = []
    text_metadata: dict[str, Any] = {
        "candidate_indices": [],
        "candidate_noun_count": 0,
        "selected_noun_examples": [],
    }
    train_image_count = int(train_dataset.num_samples)
    train_labels_available = train_dataset.labels is not None
    efficiency_measurement = None

    if stage1_load_checkpoint and stage1_checkpoint_path.exists():
        progress.log(f"Loading SEIC Stage 1 checkpoint and skipping head training: {stage1_checkpoint_path}")
        checkpoint_payload = _load_seic_stage1_checkpoint(
            heads=heads,
            checkpoint_path=stage1_checkpoint_path,
            device=device,
        )
        stage1_history = checkpoint_payload.get("stage1_history", checkpoint_payload.get("history", []))
        text_metadata = _normalize_seic_text_metadata(checkpoint_payload.get("text_metadata", text_metadata))
        stage1_checkpoint_loaded = True
    else:
        with progress.stage(f"Encoding source-train images '{train_dataset.name}/{train_split}'"):
            train_image_features, train_labels = _encode_normalized_images(
                train_dataset,
                bundle=bundle,
                batch_size=resolved["image_batch_size"],
                num_workers=inputs.runtime_config.num_workers,
                device=device,
            )
        if train_image_features.shape[1] != eval_image_features.shape[1]:
            raise ValueError(
                f"SEIC source/eval feature dimension mismatch: "
                f"{train_image_features.shape[1]} vs {eval_image_features.shape[1]}."
            )
        train_image_count = int(train_image_features.shape[0])
        train_labels_available = train_labels is not None
        with progress.stage(f"Encoding WordNet noun pool from {wordnet_csv}"):
            candidate_nouns, noun_embeddings = _encode_wordnet_nouns(
                csv_path=wordnet_csv,
                bundle=bundle,
                batch_size=resolved["text_batch_size"],
                device=device,
                cache_dir=output_dir,
                candidate_noun_limit=params.get("candidate_noun_limit"),
                force_recompute=bool(params.get("force_recompute_nouns", False)),
            )
        with progress.stage(f"Building SEIC image-text pairs with k1={resolved['k1']}, k2={resolved['k2']}"):
            train_text_features, text_metadata = _build_seic_text_features(
                image_features=train_image_features,
                noun_embeddings=noun_embeddings,
                cluster_num=cluster_num,
                k1=resolved["k1"],
                k2=resolved["k2"],
                text_weight_temperature=resolved["text_weight_temperature"],
                text_feature_chunk_size=resolved["text_feature_chunk_size"],
                random_state=None if random_state is None else int(random_state),
                cache_path=cache_file_path(output_dir, f"{text_cache_key}__text_pairs", ".npz"),
                progress=progress,
            )
        text_metadata = _normalize_seic_text_metadata(
            {
                **text_metadata,
                "selected_noun_examples": [
                    candidate_nouns[index] for index in text_metadata["candidate_indices"][:10]
                ],
            }
        )
        efficiency_measurement = start_train_eval_measurement(device)
        with progress.stage("Training SEIC Stage 1 cross-modal semantic consistency"):
            stage1_history = _train_stage1_alignment(
                heads=heads,
                image_features=train_image_features,
                text_features=train_text_features,
                params=resolved,
                device=device,
                progress=progress,
            )
        if stage1_save_checkpoint:
            stage1_checkpoint_path = checkpoint_file(output_dir, checkpoint_name, create=True)
            _save_seic_stage1_checkpoint(
                heads=heads,
                checkpoint_path=stage1_checkpoint_path,
                params=resolved,
                cluster_num=cluster_num,
                in_dim=int(train_image_features.shape[1]),
                text_cache_key=text_cache_key,
                text_metadata=text_metadata,
                stage1_history=stage1_history,
            )

    with progress.stage(f"Inferring SEIC Stage 1 predictions on '{eval_dataset.name}/{eval_split}'"):
        stage1_predictions = _predict_from_features(
            heads=heads,
            features=eval_image_features,
            batch_size=resolved["eval_batch_size"],
            device=device,
        )

    final_predictions = stage1_predictions
    stage2_history: list[dict[str, float]] = []
    lora_metadata: dict[str, Any] = {"enabled": False}
    if stage2_enabled:
        with progress.stage("Running SEIC Stage 2 self-enhanced LoRA fine-tuning"):
            final_predictions, stage2_history, lora_metadata = _train_stage2_self_enhancement(
                bundle=bundle,
                heads=heads,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                params=resolved,
                num_workers=inputs.runtime_config.num_workers,
                device=device,
                progress=progress,
            )
    if efficiency_measurement is not None:
        efficiency_measurement.stop()

    return SEICOutputs(
        predictions=final_predictions.astype(np.int64).tolist(),
        evaluation_labels=eval_labels.astype(np.int64).tolist() if eval_labels is not None else None,
        evaluation_split=eval_split,
        metadata={
            "variant": "seic" if stage2_enabled else "seic_stage1",
            "source_dataset": train_dataset.name,
            "evaluation_dataset": eval_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": eval_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "wordnet_csv": str(wordnet_csv),
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "n_clusters": cluster_num,
            "n_clusters_source": cluster_num_source,
            "k1": resolved["k1"],
            "k2": resolved["k2"],
            "text_weight_temperature": resolved["text_weight_temperature"],
            "selected_candidate_noun_count": int(text_metadata["candidate_noun_count"]),
            "selected_noun_examples": text_metadata["selected_noun_examples"],
            "train_image_count": train_image_count,
            "eval_image_count": int(eval_image_features.shape[0]),
            "train_labels_available": train_labels_available,
            "stage1": {
                "epochs": resolved["stage1_epochs"],
                "batch_size": resolved["stage1_batch_size"],
                "learning_rate": resolved["stage1_learning_rate"],
                "alpha": resolved["alpha"],
                "beta": resolved["beta"],
                "gamma": resolved["gamma"],
                "delta": resolved["delta"],
                "assignment_temperature": resolved["assignment_temperature"],
                "center_temperature": resolved["center_temperature"],
                "save_checkpoint": stage1_save_checkpoint,
                "load_checkpoint": stage1_load_checkpoint,
                "checkpoint_loaded": stage1_checkpoint_loaded,
                "checkpoint_path": str(stage1_checkpoint_path)
                if (stage1_checkpoint_loaded or stage1_save_checkpoint)
                else None,
                "training_history": stage1_history,
            },
            "stage2": {
                "enabled": stage2_enabled,
                "epochs": resolved["stage2_epochs"],
                "batch_size": resolved["stage2_batch_size"],
                "learning_rate": resolved["stage2_learning_rate"],
                "lora": lora_metadata,
                "training_history": stage2_history,
            },
            "extra_predictions": {
                "stage1": stage1_predictions.astype(np.int64).tolist(),
            },
        },
    )


def _resolve_seic_params(
    params: dict[str, Any],
    dataset_name: str | None = None,
    *,
    openclip_pretraining: str | None = None,
    openclip_backbone: str | None = None,
) -> dict[str, Any]:
    hyperparameter_defaults = {}
    if dataset_name is not None:
        hyperparameter_defaults = _resolve_seic_hyperparameter_defaults(
            openclip_pretraining or str(params.get("openclip_pretraining", "LAION400M")),
            openclip_backbone or str(params.get("openclip_backbone", "ViT-B/32")),
            dataset_name,
        )
    stage1_only = _param_bool(params.get("stage1_only"), False)
    return {
        "image_batch_size": int(params.get("image_batch_size", 256)),
        "text_batch_size": int(params.get("text_batch_size", 2048)),
        "eval_batch_size": int(params.get("eval_batch_size", params.get("image_batch_size", 256))),
        "k1": int(params.get("k1", params.get("nouns_per_initial_center", 200))),
        "k2": int(params.get("k2", params.get("nouns_per_image", 50))),
        "text_weight_temperature": float(
            params.get(
                "text_weight_temperature",
                params.get(
                    "text_temp",
                    params.get(
                        "retrieval_temperature",
                        hyperparameter_defaults.get("text_weight_temperature", 0.01),
                    ),
                ),
            )
        ),
        "text_feature_chunk_size": int(params.get("text_feature_chunk_size", params.get("text_pair_chunk_size", 4096))),
        "stage1_epochs": int(params.get("stage1_epochs", params.get("alignment_epochs", 200))),
        "stage1_batch_size": int(params.get("stage1_batch_size", params.get("alignment_batch_size", 1024))),
        "stage1_learning_rate": float(params.get("stage1_learning_rate", params.get("alignment_lr", 0.005))),
        "stage1_weight_decay": float(params.get("stage1_weight_decay", 0.0)),
        "instance_temperature": float(params.get("instance_temperature", 0.07)),
        "assignment_temperature": float(
            params.get(
                "assignment_temperature",
                params.get(
                    "assign_temp",
                    params.get(
                        "cross_modal_temperature",
                        hyperparameter_defaults.get("assignment_temperature", 0.5),
                    ),
                ),
            )
        ),
        "center_temperature": float(
            params.get(
                "center_temperature",
                params.get(
                    "center_temp",
                    params.get(
                        "cross_modal_temperature",
                        hyperparameter_defaults.get("center_temperature", 0.5),
                    ),
                ),
            )
        ),
        "balance_momentum": float(params.get("balance_momentum", 0.9)),
        "balance_history_floor": float(params.get("balance_history_floor", 0.01)),
        "balance_max_weight": float(params.get("balance_max_weight", 5.0)),
        "alpha": float(params.get("alpha", 0.5)),
        "beta": float(params.get("beta", 1.0)),
        "gamma": float(params.get("gamma", 1.0)),
        "delta": float(params.get("delta", 2.0)),
        "stage2": (not stage1_only) and _param_bool(params.get("stage2", params.get("self_enhance")), False),
        "stage2_epochs": int(params.get("stage2_epochs", params.get("self_enhance_epochs", 40))),
        "stage2_batch_size": int(params.get("stage2_batch_size", params.get("self_enhance_batch_size", 128))),
        "stage2_learning_rate": float(params.get("stage2_learning_rate", params.get("self_enhance_lr", 5.0e-5))),
        "stage2_weight_decay": float(params.get("stage2_weight_decay", 0.0)),
        "lora_rank": int(params.get("lora_rank", 128)),
        "lora_alpha": float(params.get("lora_alpha", params.get("lora_rank", 128))),
        "lora_dropout": float(params.get("lora_dropout", 0.0)),
        "confidence_momentum": float(params.get("confidence_momentum", 0.999)),
        "strong_crop_scale_min": float(params.get("strong_crop_scale_min", 0.5)),
        "strong_crop_scale_max": float(params.get("strong_crop_scale_max", 1.0)),
        "horizontal_flip_p": float(params.get("horizontal_flip_p", 0.5)),
        "color_jitter_p": float(params.get("color_jitter_p", 0.8)),
        "color_jitter_strength": float(params.get("color_jitter_strength", 0.4)),
        "grayscale_p": float(params.get("grayscale_p", 0.2)),
    }


def _resolve_seic_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_seic_profile_token(openclip_pretraining),
        _normalize_seic_profile_token(openclip_backbone),
        _normalize_seic_profile_dataset(dataset_name),
    )
    return dict(SEIC_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_seic_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    profile_key = (
        _normalize_seic_profile_token(openclip_pretraining),
        _normalize_seic_profile_token(openclip_backbone),
        _normalize_seic_profile_dataset(dataset_name),
    )
    if profile_key not in SEIC_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _normalize_seic_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_seic_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in SEIC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


_SEIC_STAGE1_CHECKPOINT_PARAM_KEYS = (
    "stage1_epochs",
    "stage1_batch_size",
    "stage1_learning_rate",
    "stage1_weight_decay",
    "instance_temperature",
    "assignment_temperature",
    "center_temperature",
    "balance_momentum",
    "balance_history_floor",
    "balance_max_weight",
    "alpha",
    "beta",
    "gamma",
    "delta",
)


def _seic_stage1_checkpoint_name(*, text_cache_key: str, in_dim: int, params: dict[str, Any]) -> str:
    stage1_key = cache_key(
        f"dim{int(in_dim)}",
        *(f"{key}{params[key]}" for key in _SEIC_STAGE1_CHECKPOINT_PARAM_KEYS),
    )
    stage1_digest = hashlib.sha1(stage1_key.encode("utf-8")).hexdigest()[:12]
    return f"{cache_key(text_cache_key, f'stage1{stage1_digest}')}__seic_stage1.pt"


def _normalize_seic_text_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = {} if metadata is None else metadata
    candidate_indices = [int(index) for index in metadata.get("candidate_indices", [])]
    return {
        "candidate_indices": candidate_indices,
        "candidate_noun_count": int(metadata.get("candidate_noun_count", len(candidate_indices))),
        "selected_noun_examples": [str(noun) for noun in metadata.get("selected_noun_examples", [])],
    }


def _save_seic_stage1_checkpoint(
    *,
    heads: SEICHeads,
    checkpoint_path: Path,
    params: dict[str, Any],
    cluster_num: int,
    in_dim: int,
    text_cache_key: str,
    text_metadata: dict[str, Any],
    stage1_history: list[dict[str, float]],
) -> None:
    torch = require_module("torch", "pip install torch")
    torch.save(
        {
            "heads": heads.state_dict(),
            "cluster_num": int(cluster_num),
            "in_dim": int(in_dim),
            "text_cache_key": text_cache_key,
            "stage1_params": {key: params[key] for key in _SEIC_STAGE1_CHECKPOINT_PARAM_KEYS},
            "text_metadata": _normalize_seic_text_metadata(text_metadata),
            "stage1_history": stage1_history,
        },
        checkpoint_path,
    )


def _load_seic_stage1_checkpoint(*, heads: SEICHeads, checkpoint_path: Path, device) -> dict[str, Any]:
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    heads.load_state_dict(payload["heads"], strict=True)
    return payload


def _param_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _encode_normalized_images(
    dataset: LoadedImageDataset,
    *,
    bundle: OpenCLIPBundle,
    batch_size: int,
    num_workers: int,
    device,
) -> tuple[np.ndarray, np.ndarray | None]:
    raw_features, labels = load_or_compute_raw_image_embeddings(
        dataset=dataset,
        bundle=bundle,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    return _normalize_rows(raw_features.astype("float32")), labels


def _encode_wordnet_nouns(
    *,
    csv_path: Path,
    bundle: OpenCLIPBundle,
    batch_size: int,
    device,
    cache_dir: Path,
    candidate_noun_limit: Any,
    force_recompute: bool,
) -> tuple[list[str], np.ndarray]:
    del cache_dir
    limit_value = None if candidate_noun_limit in {None, "", "none"} else int(candidate_noun_limit)
    all_nouns = _load_tac_wordnet_nouns(csv_path, None)
    _, _, ensemble_embeddings = load_or_compute_text_prompt_bank(
        csv_path=csv_path,
        nouns=all_nouns,
        bundle=bundle,
        batch_size=batch_size,
        device=device,
        cache_dir=common_feature_cache_dir(csv_path.parent, "text"),
        prompt_builders=SEIC_WORDNET_PROMPT_BUILDERS,
        template_key=SEIC_WORDNET_PROMPT_TEMPLATE_KEY,
        normalize_prompt_embeddings=True,
        normalize_ensemble=True,
        force_recompute=force_recompute,
    )
    if limit_value is None:
        return all_nouns, ensemble_embeddings
    return all_nouns[:limit_value], ensemble_embeddings[:limit_value]


def _build_seic_text_features(
    *,
    image_features: np.ndarray,
    noun_embeddings: np.ndarray,
    cluster_num: int,
    k1: int,
    k2: int,
    text_weight_temperature: float,
    text_feature_chunk_size: int,
    random_state: int | None,
    cache_path: Path,
    progress=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return payload["text_features"], {
            "candidate_indices": [int(index) for index in payload["candidate_indices"].tolist()],
            "candidate_noun_count": int(payload["candidate_noun_count"][0]),
        }

    if k1 <= 0 or k2 <= 0:
        raise ValueError("SEIC requires k1 and k2 to be positive.")
    if text_weight_temperature <= 0:
        raise ValueError("SEIC text_weight_temperature must be positive.")
    if text_feature_chunk_size <= 0:
        raise ValueError("SEIC text_feature_chunk_size must be positive.")

    image_features = _normalize_rows(image_features.astype("float32", copy=False))
    noun_embeddings = _normalize_rows(noun_embeddings.astype("float32", copy=False))
    assignments, initial_centers = run_faiss_kmeans(
        image_features,
        n_clusters=cluster_num,
        n_iter=300,
        n_redo=10,
        spherical=True,
        random_state=random_state,
        progress=progress,
        label="SEIC initial image KMeans",
    )
    del assignments
    initial_centers = _normalize_rows(initial_centers.astype("float32", copy=False))

    effective_k1 = min(int(k1), int(noun_embeddings.shape[0]))
    _, center_noun_indices = search_topk(
        noun_embeddings,
        topk=effective_k1,
        faiss_metric="ip",
        query_features=initial_centers,
        sklearn_metric="cosine",
        progress=progress,
        label="SEIC center-to-noun retrieval",
    )
    candidate_indices = np.unique(center_noun_indices.reshape(-1)).astype(np.int64)
    candidate_embeddings = noun_embeddings[candidate_indices]
    effective_k2 = min(int(k2), int(candidate_embeddings.shape[0]))
    similarity, local_indices = search_topk(
        candidate_embeddings,
        topk=effective_k2,
        faiss_metric="ip",
        query_features=image_features,
        sklearn_metric="cosine",
        progress=progress,
        label="SEIC image-to-candidate-noun retrieval",
    )

    text_features = _compute_weighted_text_features(
        candidate_embeddings=candidate_embeddings,
        local_indices=local_indices,
        similarity=similarity,
        temperature=float(text_weight_temperature),
        chunk_size=int(text_feature_chunk_size),
        progress=progress,
    )
    np.savez_compressed(
        cache_path,
        text_features=text_features,
        candidate_indices=candidate_indices,
        candidate_noun_count=np.asarray([candidate_indices.shape[0]], dtype=np.int64),
        k1=np.asarray([k1], dtype=np.int64),
        k2=np.asarray([k2], dtype=np.int64),
    )
    return text_features, {
        "candidate_indices": [int(index) for index in candidate_indices.tolist()],
        "candidate_noun_count": int(candidate_indices.shape[0]),
    }


def _compute_weighted_text_features(
    *,
    candidate_embeddings: np.ndarray,
    local_indices: np.ndarray,
    similarity: np.ndarray,
    temperature: float,
    chunk_size: int,
    progress=None,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("SEIC text_weight_temperature must be positive.")
    if chunk_size <= 0:
        raise ValueError("SEIC text_feature_chunk_size must be positive.")
    if local_indices.shape != similarity.shape:
        raise ValueError(f"SEIC top-k index/similarity shape mismatch: {local_indices.shape} vs {similarity.shape}.")

    row_count = int(local_indices.shape[0])
    text_features = np.empty((row_count, candidate_embeddings.shape[1]), dtype=np.float32)
    if row_count == 0:
        return text_features

    total_chunks = (row_count + int(chunk_size) - 1) // int(chunk_size)
    if progress is not None and total_chunks > 1:
        progress.log(
            f"Computing SEIC weighted text features in {total_chunks} chunk(s) "
            f"of up to {int(chunk_size)} sample(s)"
        )

    for chunk_index, start in enumerate(range(0, row_count, int(chunk_size)), start=1):
        end = min(start + int(chunk_size), row_count)
        chunk_weights = _softmax_numpy(similarity[start:end] / float(temperature), axis=1).astype(
            "float32",
            copy=False,
        )
        chunk_embeddings = candidate_embeddings[local_indices[start:end]]
        text_features[start:end] = np.einsum(
            "nk,nkd->nd",
            chunk_weights,
            chunk_embeddings,
            optimize=True,
        ).astype("float32", copy=False)
        if progress is not None:
            progress.step("SEIC weighted text feature construction", chunk_index, total_chunks, noun="chunk")

    return _normalize_rows_inplace(text_features)


def _train_stage1_alignment(
    *,
    heads: SEICHeads,
    image_features: np.ndarray,
    text_features: np.ndarray,
    params: dict[str, Any],
    device,
    progress,
) -> list[dict[str, float]]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch")

    if image_features.shape != text_features.shape:
        raise ValueError(f"SEIC image/text feature shape mismatch: {image_features.shape} vs {text_features.shape}.")
    if image_features.shape[0] < 2:
        raise ValueError("SEIC Stage 1 requires at least two source-train samples.")

    image_tensor = torch.from_numpy(image_features.astype("float32"))
    text_tensor = torch.from_numpy(text_features.astype("float32"))
    dataset = data_mod.TensorDataset(image_tensor, text_tensor)
    requested_batch_size = int(params["stage1_batch_size"])
    effective_batch_size = min(requested_batch_size, len(dataset))
    if effective_batch_size != requested_batch_size:
        progress.log(
            f"Reducing SEIC Stage 1 batch size from {requested_batch_size} to {effective_batch_size} "
            f"for {len(dataset)} training sample(s)"
        )
    drop_last = len(dataset) > effective_batch_size and len(dataset) % effective_batch_size == 1
    loader = data_mod.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=drop_last)

    optimizer = torch.optim.Adam(
        heads.parameters(),
        lr=float(params["stage1_learning_rate"]),
        weight_decay=float(params["stage1_weight_decay"]),
    )
    assignment_history = torch.full(
        (heads.num_clusters,),
        1.0 / float(heads.num_clusters),
        dtype=torch.float32,
        device=device,
    )
    history: list[dict[str, float]] = []
    epochs = int(params["stage1_epochs"])
    progress.log(f"Training SEIC Stage 1 heads for {epochs} epoch(s)")
    for epoch in range(epochs):
        heads.train()
        totals = {"loss": 0.0, "instance": 0.0, "assignment": 0.0, "center": 0.0, "balance": 0.0}
        steps = 0
        for batch_image, batch_text in loader:
            batch_image = batch_image.to(device)
            batch_text = batch_text.to(device)
            outputs = heads.forward_pair(batch_image, batch_text)
            losses = alignment_loss(
                image_projected=outputs["image_projected"],
                text_projected=outputs["text_projected"],
                image_probabilities=outputs["image_probabilities"],
                text_probabilities=outputs["text_probabilities"],
                assignment_history=assignment_history,
                instance_logit_scale=heads.instance_logit_scale,
                assignment_temperature=float(params["assignment_temperature"]),
                center_temperature=float(params["center_temperature"]),
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                gamma=float(params["gamma"]),
                delta=float(params["delta"]),
                balance_history_floor=float(params["balance_history_floor"]),
                balance_max_weight=float(params["balance_max_weight"]),
            )
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            with torch.no_grad():
                batch_assignments = torch.argmax(outputs["image_probabilities"], dim=1)
                batch_histogram = torch.bincount(batch_assignments, minlength=heads.num_clusters).float()
                batch_histogram = batch_histogram / torch.clamp(batch_histogram.sum(), min=1.0)
                momentum = float(params["balance_momentum"])
                assignment_history.mul_(momentum).add_(batch_histogram * (1.0 - momentum))
                assignment_history.div_(torch.clamp(assignment_history.sum(), min=1.0e-8))

            totals["loss"] += float(losses.total.item())
            totals["instance"] += float(losses.instance.item())
            totals["assignment"] += float(losses.assignment.item())
            totals["center"] += float(losses.center.item())
            totals["balance"] += float(losses.balance.item())
            steps += 1

        record = {name: value / max(steps, 1) for name, value in totals.items()}
        record["epoch"] = float(epoch + 1)
        record["assignment_history_std"] = float(assignment_history.std().item())
        record["instance_logit_scale"] = float(torch.clamp(heads.instance_logit_scale.exp(), max=100.0).item())
        history.append(record)
        progress.epoch(
            "SEIC Stage 1",
            epoch + 1,
            epochs,
            metrics={
                "loss": record["loss"],
                "ins": record["instance"],
                "ass": record["assignment"],
                "ctr": record["center"],
                "bal": record["balance"],
            },
        )
    return history


def _train_stage2_self_enhancement(
    *,
    bundle: OpenCLIPBundle,
    heads: SEICHeads,
    train_dataset: LoadedImageDataset,
    eval_dataset: LoadedImageDataset,
    params: dict[str, Any],
    num_workers: int,
    device,
    progress,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")

    heads.set_stage2_trainable()
    lora_result = inject_lora_into_visual_attention(
        bundle.model.visual,
        rank=int(params["lora_rank"]),
        alpha=float(params["lora_alpha"]),
        dropout=float(params["lora_dropout"]),
    )
    model = SEICVisionModel(bundle.model, heads).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("SEIC Stage 2 found no trainable parameters after LoRA injection.")

    image_size = _resolve_openclip_image_size(bundle)
    mean, std = _resolve_openclip_normalization(bundle)
    weak_transform, strong_transform = _build_stage2_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        crop_scale=(float(params["strong_crop_scale_min"]), float(params["strong_crop_scale_max"])),
        horizontal_flip_p=float(params["horizontal_flip_p"]),
        color_jitter_p=float(params["color_jitter_p"]),
        color_jitter_strength=float(params["color_jitter_strength"]),
        grayscale_p=float(params["grayscale_p"]),
    )
    train_view = SEICSelfEnhancementDataset(train_dataset, weak_transform, strong_transform)
    eval_view = SEICEvalDataset(eval_dataset, weak_transform)
    if len(train_view) < 2:
        raise ValueError("SEIC Stage 2 requires at least two source-train samples.")

    requested_batch_size = int(params["stage2_batch_size"])
    effective_batch_size = min(requested_batch_size, len(train_view))
    if effective_batch_size != requested_batch_size:
        progress.log(
            f"Reducing SEIC Stage 2 batch size from {requested_batch_size} to {effective_batch_size} "
            f"for {len(train_view)} training sample(s)"
        )
    train_loader = data_mod.DataLoader(
        train_view,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    eval_loader = data_mod.DataLoader(
        eval_view,
        batch_size=int(params["eval_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(params["stage2_learning_rate"]),
        weight_decay=float(params["stage2_weight_decay"]),
    )
    confidence_stats = _ConfidenceStats(
        momentum=float(params["confidence_momentum"]),
    )
    history: list[dict[str, float]] = []
    epochs = int(params["stage2_epochs"])
    progress.log(
        f"Training SEIC Stage 2 for {epochs} epoch(s) with {lora_result.modules_replaced} LoRA attention module(s)"
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_confidence = 0.0
        total_weight = 0.0
        steps = 0
        samples = 0
        for batch in train_loader:
            weak = batch["weak"].to(device, non_blocking=True)
            strong = batch["strong"].to(device, non_blocking=True)
            with torch.no_grad():
                _, weak_logits, weak_probabilities = model(weak)
                del weak_logits
                confidence, pseudo_labels = torch.max(weak_probabilities, dim=1)
                confidence_stats.update(confidence.detach())
                weights = confidence_stats.weights(confidence)
            _, strong_logits, _ = model(strong)
            loss = self_enhancement_loss(strong_logits, pseudo_labels, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(weak.size(0))
            total_loss += float(loss.item())
            total_confidence += float(confidence.mean().item()) * batch_size
            total_weight += float(weights.mean().item()) * batch_size
            samples += batch_size
            steps += 1

        record = {
            "epoch": float(epoch + 1),
            "loss": total_loss / max(steps, 1),
            "mean_confidence": total_confidence / max(samples, 1),
            "mean_weight": total_weight / max(samples, 1),
            "confidence_ema_mean": confidence_stats.mean,
            "confidence_ema_var": confidence_stats.var,
        }
        history.append(record)
        progress.epoch(
            "SEIC Stage 2",
            epoch + 1,
            epochs,
            metrics={
                "loss": record["loss"],
                "conf": record["mean_confidence"],
                "weight": record["mean_weight"],
            },
        )

    progress.log("Inferring SEIC final predictions with the self-enhanced visual encoder")
    predictions, _labels = _predict_from_images(model, eval_loader, device)
    return predictions, history, {
        "enabled": True,
        "rank": int(params["lora_rank"]),
        "alpha": float(params["lora_alpha"]),
        "dropout": float(params["lora_dropout"]),
        "modules_replaced": int(lora_result.modules_replaced),
        "trainable_visual_parameters": int(lora_result.trainable_parameters),
        "trainable_total_parameters": int(sum(parameter.numel() for parameter in trainable_params)),
    }


class _ConfidenceStats:
    def __init__(self, *, momentum: float) -> None:
        self.mean: float | None = None
        self.var: float | None = None
        self.momentum = float(momentum)

    def update(self, confidence) -> None:
        batch_mean = float(confidence.mean().item())
        batch_var = float(confidence.var(unbiased=False).item())
        batch_var = max(batch_var, 1.0e-6)
        if self.mean is None or self.var is None:
            self.mean = batch_mean
            self.var = batch_var
            return
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var

    def weights(self, confidence):
        torch = require_module("torch", "pip install torch")

        if self.mean is None or self.var is None:
            self.update(confidence.detach())
        mean = torch.as_tensor(self.mean, device=confidence.device, dtype=confidence.dtype)
        var = torch.as_tensor(max(float(self.var), 1.0e-6), device=confidence.device, dtype=confidence.dtype)
        below_mean = confidence < mean
        gaussian = torch.exp(-((confidence - mean) ** 2) / (2.0 * var))
        return torch.where(below_mean, gaussian, torch.ones_like(confidence))


def _predict_from_features(*, heads: SEICHeads, features: np.ndarray, batch_size: int, device) -> np.ndarray:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch")

    loader = data_mod.DataLoader(
        data_mod.TensorDataset(torch.from_numpy(features.astype("float32"))),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    heads.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device)
            _, _logits, probabilities = heads.forward_image_features(batch_features)
            predictions.append(torch.argmax(probabilities, dim=1).cpu().numpy().astype(np.int64))
    return np.concatenate(predictions, axis=0)


def _predict_from_images(model: SEICVisionModel, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    torch = require_module("torch", "pip install torch")

    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            _, _logits, probabilities = model(images)
            predictions.append(torch.argmax(probabilities, dim=1).cpu().numpy().astype(np.int64))
            labels.append(batch["target"].numpy().astype(np.int64))
    return np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)


def _build_stage2_transforms(
    *,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_scale: tuple[float, float],
    horizontal_flip_p: float,
    color_jitter_p: float,
    color_jitter_strength: float,
    grayscale_p: float,
):
    transforms = require_module("torchvision.transforms", "pip install torch torchvision")
    weak_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    strong_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=crop_scale),
            transforms.RandomHorizontalFlip(p=horizontal_flip_p),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.8 * color_jitter_strength,
                        contrast=0.8 * color_jitter_strength,
                        saturation=0.8 * color_jitter_strength,
                        hue=0.2 * color_jitter_strength,
                    )
                ],
                p=color_jitter_p,
            ),
            transforms.RandomGrayscale(p=grayscale_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return weak_transform, strong_transform


def _resolve_openclip_image_size(bundle: OpenCLIPBundle) -> int:
    image_size = getattr(bundle.model.visual, "image_size", 224)
    if isinstance(image_size, tuple):
        return int(image_size[0])
    return int(image_size)


def _resolve_openclip_normalization(bundle: OpenCLIPBundle) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    transforms = getattr(bundle.preprocess, "transforms", [])
    tv_transforms = require_module("torchvision.transforms", "pip install torch torchvision")
    for transform in transforms:
        if isinstance(transform, tv_transforms.Normalize):
            return tuple(float(value) for value in transform.mean), tuple(float(value) for value in transform.std)
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1.0e-12, None)


def _normalize_rows_inplace(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, np.clip(norms, 1.0e-12, None), out=matrix)
    return matrix


def _softmax_numpy(matrix: np.ndarray, axis: int) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=axis, keepdims=True), 1.0e-12, None)
