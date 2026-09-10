from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_dir as ensure_checkpoint_dir
from vlm4cluster.methods.checkpointing import checkpoint_interval, save_checkpoint_enabled
from vlm4cluster.methods.checkpointing import first_existing_checkpoint, latest_checkpoint
from vlm4cluster.methods.checkpointing import load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.cluster_defaults import resolve_source_default_out_dim
from vlm4cluster.methods.feature_cache import (
    load_or_compute_raw_image_embeddings,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
)
from vlm4cluster.methods.tac.core import _normalize_dataset_name, _resolve_domain_shift_datasets
from vlm4cluster.models import load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.io import ensure_dir
from vlm4cluster.utils.progress import get_progress_logger


TEMI_TABLE10_CLUSTERING_DEFAULTS: dict[str, Any] = {
    "knn": 50,
    "num_heads": 50,
    "epochs": 200,
    "batch_size": 512,
    "optimizer": "adamw",
    "learning_rate": 1.0e-4,
    "min_learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "weight_decay_end": 1.0e-4,
    "warmup_epochs": 20,
    "momentum_teacher": 0.996,
    "max_momentum_teacher": 0.996,
    "beta": 0.6,
    "embed_aug": "none",
    "num_augs": 1,
}


TEMI_SMALL_MEDIUM_DATASET_DEFAULTS: dict[str, Any] = {
    **TEMI_TABLE10_CLUSTERING_DEFAULTS,
}


TEMI_LARGE_SCALE_DATASET_DEFAULTS: dict[str, Any] = {
    **TEMI_TABLE10_CLUSTERING_DEFAULTS,
    "knn": 25,
    "batch_size": 1024,
    "warmup_epochs": 10,
    "knn_chunk_size": 512,
}


TEMI_DATASET_HYPERPARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "cifar10": {**TEMI_TABLE10_CLUSTERING_DEFAULTS},
    "cifar100": {**TEMI_TABLE10_CLUSTERING_DEFAULTS},
    "cifar20": {**TEMI_TABLE10_CLUSTERING_DEFAULTS, "beta": 0.55},
    "stl10": {**TEMI_TABLE10_CLUSTERING_DEFAULTS, "epochs": 800},
    "dtd": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS, "knn": 20, "beta": 0.7},
    "aircraft": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS, "knn": 20, "beta": 0.7},
    "cars": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS, "knn": 20, "beta": 0.65},
    "flowers": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS},
    "food": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS},
    "pets": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS, "knn": 20, "beta": 0.75},
    "ucf101": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS},
    "imagenet10": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS},
    "imagenetdogs": {**TEMI_SMALL_MEDIUM_DATASET_DEFAULTS},
    "imagenet": {
        **TEMI_TABLE10_CLUSTERING_DEFAULTS,
        "knn": 25,
        "batch_size": 1024,
        "warmup_epochs": 10,
    },
    "places365standard": {**TEMI_LARGE_SCALE_DATASET_DEFAULTS},
}

TEMI_BACKBONE_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb16", "aircraft"): {"teacher_temp": 0.05},
    ("laion400m", "vitb16", "flowers"): {"teacher_temp": 0.02, "knn": 25},
    ("laion400m", "vitb16", "pets"): {"teacher_temp": 0.15, "knn": 25},
    ("laion400m", "vitl14", "dtd"): {"teacher_temp": 0.05},
    ("laion400m", "vitl14", "flowers"): {"teacher_temp": 0.02, "knn": 25},
}

TEMI_GENERIC_OPENCLIP_CACHE_TOKEN = "generic-openclip-defaults-v1"


@dataclass(slots=True)
class TEMIOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class DINOHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_bn: bool = False,
        dropout_p: float = 0.0,
        final_gelu: bool = False,
        norm_last_layer: bool = False,
        nlayers: int = 2,
        hidden_dim: int = 512,
        bottleneck_dim: int = 256,
    ) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            layers: list[object] = [torch.nn.Linear(in_dim, bottleneck_dim)]
            if final_gelu:
                layers.append(torch.nn.GELU())
        else:
            layers = [torch.nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.GELU())
            if dropout_p > 0:
                layers.append(torch.nn.Dropout(dropout_p))
            for _ in range(nlayers - 2):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(torch.nn.BatchNorm1d(hidden_dim))
                layers.append(torch.nn.GELU())
                if dropout_p > 0:
                    layers.append(torch.nn.Dropout(dropout_p))
            layers.append(torch.nn.Linear(hidden_dim, bottleneck_dim))
            if final_gelu:
                layers.append(torch.nn.GELU())

        self.mlp = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, module) -> None:
        torch = require_module("torch", "pip install torch")
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        torch = require_module("torch", "pip install torch")
        x = self.mlp(x)
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, head_args: dict[str, Any], num_heads: int) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        if num_heads < 1:
            raise ValueError("TEMI requires num_heads >= 1.")
        self.num_heads = num_heads
        self.heads = torch.nn.ModuleList([DINOHead(**head_args) for _ in range(num_heads)])
        self.register_buffer("best_head_idx", torch.tensor(0, dtype=torch.long))

    @property
    def best_head(self):
        return self.heads[int(self.best_head_idx.item())]

    def set_losses(self, losses) -> None:
        if self.num_heads == 1:
            return
        if len(losses) != self.num_heads:
            raise ValueError("Number of TEMI head losses does not match num_heads.")
        self.best_head_idx = require_module("torch", "pip install torch").argmin(losses.detach()).to(
            self.best_head_idx.device
        )

    def forward(self, x):
        if not self.training or self.num_heads == 1:
            return self.best_head(x)
        return [head(x) for head in self.heads]


class TEMIHeadModel(require_module("torch.nn", "pip install torch").Module):
    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        *,
        num_heads: int,
        use_bn_in_head: bool,
        head_dropout_prob: float,
        head_final_gelu: bool,
        norm_last_layer: bool,
        nlayers: int,
        hidden_dim: int,
        bottleneck_dim: int,
        l2_norm: bool,
        embed_norm: bool,
        embed_mean,
        embed_std,
    ) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.l2_norm = l2_norm
        self.embed_norm = embed_norm
        self.register_buffer("embed_mean", embed_mean.clone().detach())
        self.register_buffer("embed_std", embed_std.clone().detach())
        self.head = MultiHead(
            {
                "in_dim": embed_dim,
                "out_dim": out_dim,
                "use_bn": use_bn_in_head,
                "dropout_p": head_dropout_prob,
                "final_gelu": head_final_gelu,
                "norm_last_layer": norm_last_layer,
                "nlayers": nlayers,
                "hidden_dim": hidden_dim,
                "bottleneck_dim": bottleneck_dim,
            },
            num_heads=num_heads,
        )

    def embed(self, x):
        torch = require_module("torch", "pip install torch")
        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        if self.embed_norm:
            x = (x - self.embed_mean) / self.embed_std
        if self.l2_norm:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return x

    def forward(self, x):
        return self.head(self.embed(x))


class TEMIStudentTeacher(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, student: TEMIHeadModel, teacher: TEMIHeadModel) -> None:
        super().__init__()
        teacher.load_state_dict(student.state_dict())
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        self.student = student
        self.teacher = teacher

    def forward(self, views):
        return self.teacher(views), self.student(views)


class TEMILoss(require_module("torch.nn", "pip install torch").Module):
    def __init__(
        self,
        *,
        out_dim: int,
        batch_size: int,
        num_heads: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        epochs: int,
        student_temp: float,
        beta: float,
        probs_momentum: float,
        positive_pmi: bool,
    ) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.student_temp = student_temp
        self.beta = beta
        self.probs_momentum = probs_momentum
        self.min_mi = 0.0 if positive_pmi else -torch.inf
        schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(max(epochs - warmup_teacher_temp_epochs, 0)) * teacher_temp,
            )
        )
        if schedule.size < epochs:
            schedule = np.pad(schedule, (0, epochs - schedule.size), constant_values=teacher_temp)
        self.register_buffer("teacher_temp_schedule", torch.tensor(schedule[:epochs], dtype=torch.float32))
        self.register_buffer("pk", torch.ones(num_heads, out_dim) / out_dim)

    @property
    def pos_probs(self):
        return self.pk[0]

    def student_probs(self, x):
        torch = require_module("torch", "pip install torch")
        return torch.nn.functional.softmax(x / self.student_temp, dim=-1)

    def teacher_probs(self, x, epoch: int):
        torch = require_module("torch", "pip install torch")
        temperature = self.teacher_temp_schedule[min(epoch, self.teacher_temp_schedule.numel() - 1)]
        return torch.nn.functional.softmax(x / temperature, dim=-1)

    def forward(self, student_out, teacher_out, epoch: int):
        torch = require_module("torch", "pip install torch")
        if isinstance(student_out, torch.Tensor):
            student_out = [student_out]
        if isinstance(teacher_out, torch.Tensor):
            teacher_out = [teacher_out]
        weights = self.compute_weight(teacher_out, epoch)
        multi_loss = []
        for index, (student_logits, teacher_logits) in enumerate(zip(student_out, teacher_out)):
            ncrops_student = len(student_logits) // self.batch_size
            ncrops_teacher = len(teacher_logits) // self.batch_size
            student_probs = self.student_probs(student_logits).chunk(ncrops_student)
            with torch.no_grad():
                teacher_probs = self.teacher_probs(teacher_logits, epoch).detach().chunk(ncrops_teacher)
            head_loss = 0.0
            count = 0
            for teacher_view in range(ncrops_teacher):
                for student_view in range(ncrops_student):
                    if student_view == teacher_view:
                        continue
                    head_loss += (
                        weights
                        * _beta_mi(
                            student_probs[student_view],
                            teacher_probs[teacher_view],
                            self.pk[index],
                            beta=self.beta,
                            clip_min=self.min_mi,
                        )
                    ).mean()
                    count += 1
            multi_loss.append(head_loss / count)
        return torch.stack(multi_loss)

    def compute_weight(self, teacher_out, epoch: int):
        torch = require_module("torch", "pip install torch")
        if isinstance(teacher_out, torch.Tensor):
            teacher_out = [teacher_out]
        weight_total = 0.0
        weight_count = 0
        with torch.no_grad():
            for index, teacher_logits in enumerate(teacher_out):
                ncrops_teacher = len(teacher_logits) // self.batch_size
                teacher_probs = self.teacher_probs(teacher_logits, epoch).detach().chunk(ncrops_teacher)
                self.pk[index] = _update_ema_probs(teacher_probs, self.pk[index], self.probs_momentum)
                for first_view in range(ncrops_teacher):
                    for second_view in list(range(ncrops_teacher))[first_view + 1 :]:
                        weight_total += _sim_weight(teacher_probs[first_view], teacher_probs[second_view])
                        weight_count += 1
        return weight_total / weight_count


class TEMIPairDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None,
        neighbors: np.ndarray,
        *,
        num_augs: int,
        embed_aug: str,
        gaussian_std: float,
    ) -> None:
        torch = require_module("torch", "pip install torch")
        self.features = torch.from_numpy(features.astype("float32"))
        self.labels = None if labels is None else torch.from_numpy(labels.astype("int64"))
        self.neighbors = neighbors.astype(np.int64)
        self.num_augs = num_augs
        self.embed_aug = embed_aug
        self.gaussian_std = gaussian_std

    def __len__(self) -> int:
        return int(self.features.size(0))

    def __getitem__(self, index: int):
        pair_index = int(np.random.choice(self.neighbors[index], 1)[0])
        views = []
        for feature in (self.features[index], self.features[pair_index]):
            for _ in range(self.num_augs):
                views.append(self._augment(feature))
        label = -1 if self.labels is None else int(self.labels[index])
        return views, label

    def _augment(self, feature):
        torch = require_module("torch", "pip install torch")
        if self.embed_aug == "none":
            return feature
        if self.embed_aug == "gaussian":
            return feature + torch.randn_like(feature) * self.gaussian_std
        raise ValueError(f"Unsupported TEMI embed_aug '{self.embed_aug}'.")


def run_temi_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> TEMIOutputs:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("temi", params)

    seed = int(params.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(
        f"Loading OpenCLIP visual backbone '{openclip_backbone}' "
        f"pretrained on '{openclip_pretraining}'"
    )
    bundle = load_openclip_bundle(
        openclip_pretraining,
        openclip_backbone,
        str(device),
    )
    cache_root = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "temi",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="TEMI",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if eval_dataset.labels is None:
        raise ValueError("TEMI requires labels on the evaluation split.")

    hyperparameter_defaults = _resolve_temi_hyperparameter_defaults(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    backbone_hyperparameter_profile = _resolve_temi_backbone_hyperparameter_profile_source(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    if hyperparameter_defaults:
        progress.log(f"Using TEMI '{_normalize_dataset_name(train_dataset.name)}' hyperparameter profile")

    out_dim, out_dim_source = resolve_source_default_out_dim(
        params,
        train_dataset,
        method_name="TEMI",
    )
    if out_dim <= 0:
        raise ValueError("TEMI requires out_dim > 0.")

    knn = int(_resolve_temi_param(params, hyperparameter_defaults, "knn", 50))
    if knn <= 0:
        raise ValueError("TEMI requires knn > 0.")

    num_heads = int(_resolve_temi_param(params, hyperparameter_defaults, "num_heads", 16))
    epochs = int(_resolve_temi_param(params, hyperparameter_defaults, "epochs", 100))
    batch_size = int(_resolve_temi_param(params, hyperparameter_defaults, "batch_size", 1024))
    if num_heads <= 0:
        raise ValueError("TEMI requires num_heads > 0.")
    if batch_size <= 0:
        raise ValueError("TEMI requires batch_size > 0.")
    requested_batch_size = batch_size

    artifact_dir = ensure_dir(
        cache_root
        / _artifact_token(
            source_dataset=train_dataset.name,
            train_split=train_split,
            evaluation_dataset=eval_dataset.name,
            evaluation_split=eval_split,
            out_dim=out_dim,
            knn=knn,
        )
    )
    knn_cache_path = artifact_dir / f"train_knn_k{knn}.npz"

    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)
    save_checkpoint_frequency = checkpoint_interval(
        params,
        enabled=save_checkpoint,
        default=20,
        aliases=("save_checkpoint_frequency", "saveckp_freq", "save_every", "checkpoint_every"),
    )
    checkpoint_artifact_dir = _resolve_temi_checkpoint_artifact_dir(
        artifact_dir,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    checkpoint_dir = ensure_checkpoint_dir(checkpoint_artifact_dir) if save_checkpoint else None
    source_artifact_dir = cache_root / _artifact_token(
        source_dataset=train_dataset.name,
        train_split=train_split,
        evaluation_dataset=train_dataset.name,
        evaluation_split="val",
        out_dim=out_dim,
        knn=knn,
    )
    source_checkpoint_artifact_dir = _resolve_temi_checkpoint_artifact_dir(
        source_artifact_dir,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    load_checkpoint = load_checkpoint_enabled(params, train_dataset.name)
    existing_checkpoint = (
        first_existing_checkpoint(
            [
                path
                for path in (
                    latest_checkpoint(checkpoint_artifact_dir / "checkpoints", "temi_epoch_*.pt"),
                    latest_checkpoint(source_checkpoint_artifact_dir / "checkpoints", "temi_epoch_*.pt"),
                )
                if path is not None
            ]
        )
        if load_checkpoint
        else None
    )

    image_batch_size = int(_resolve_temi_param(params, hyperparameter_defaults, "image_batch_size", 256))
    with progress.stage(f"Encoding evaluation image split '{eval_dataset.name}/{eval_split}'"):
        eval_features, eval_labels = load_or_compute_raw_image_embeddings(
            dataset=eval_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )

    embed_aug = str(_resolve_temi_param(params, hyperparameter_defaults, "embed_aug", "none"))
    num_augs = int(_resolve_temi_param(params, hyperparameter_defaults, "num_augs", 1))
    gaussian_std = float(_resolve_temi_param(params, hyperparameter_defaults, "gaussian_std", 0.1))
    if num_augs <= 0:
        raise ValueError("TEMI requires num_augs > 0.")

    checkpoint_payload = None
    checkpoint_loaded = False
    loaded_checkpoint_path = None
    embed_dim = int(eval_features.shape[1])
    if load_checkpoint and existing_checkpoint is not None:
        progress.log(f"Loading TEMI checkpoint and skipping train KNN/head training: {existing_checkpoint}")
        checkpoint_payload = load_torch_checkpoint(existing_checkpoint, device=device)
        checkpoint_metadata = checkpoint_payload.get("metadata", {})
        if not isinstance(checkpoint_metadata, dict):
            checkpoint_metadata = {}
        _validate_temi_checkpoint_metadata(
            checkpoint_metadata,
            out_dim=out_dim,
            num_heads=num_heads,
            checkpoint_path=existing_checkpoint,
        )
        embed_dim = int(checkpoint_metadata.get("embed_dim", embed_dim))
        if embed_dim != int(eval_features.shape[1]):
            raise ValueError(
                "TEMI checkpoint embedding dimension does not match the evaluation features: "
                f"checkpoint embed_dim={embed_dim}, eval feature dim={int(eval_features.shape[1])}. "
                f"Checkpoint: {existing_checkpoint}"
            )
        checkpoint_loaded = True
        loaded_checkpoint_path = existing_checkpoint
        progress.log("Skipping TEMI train feature encoding, train KNN graph construction, and head training")

    train_features = None
    train_loader = None
    if not checkpoint_loaded:
        with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
            train_features, train_labels = load_or_compute_raw_image_embeddings(
                dataset=train_dataset,
                bundle=bundle,
                batch_size=image_batch_size,
                num_workers=inputs.runtime_config.num_workers,
                device=device,
            )
        if train_features.shape[0] < 2:
            raise ValueError("TEMI training requires at least two training samples.")
        if knn >= train_features.shape[0]:
            raise ValueError(
                f"TEMI knn ({knn}) must be smaller than the source train size ({train_features.shape[0]})."
            )
        batch_size = min(batch_size, train_features.shape[0])
        if batch_size != requested_batch_size:
            progress.log(
                f"Reducing TEMI batch size from {requested_batch_size} to {batch_size} "
                f"for {train_features.shape[0]} training sample(s)"
            )

        with progress.stage(f"Building TEMI train KNN graph with k={knn}"):
            neighbors = _load_or_compute_train_neighbors(
                train_features,
                k=knn,
                cache_path=knn_cache_path,
                device=device,
                chunk_size=int(_resolve_temi_param(params, hyperparameter_defaults, "knn_chunk_size", 4096)),
                force_recompute=bool(params.get("force_recompute_knn", False)),
            )

        train_view = TEMIPairDataset(
            train_features,
            train_labels,
            neighbors,
            num_augs=num_augs,
            embed_aug=embed_aug,
            gaussian_std=gaussian_std,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = data_mod.DataLoader(
            train_view,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=generator,
        )
        if len(train_loader) == 0:
            raise ValueError("TEMI training loader is empty after applying drop_last=True.")
        embed_dim = int(train_features.shape[1])

    embed_norm = bool(_resolve_temi_param(params, hyperparameter_defaults, "embed_norm", False))
    if train_features is None:
        embed_mean = torch.zeros(embed_dim, dtype=torch.float32)
        embed_std = torch.ones(embed_dim, dtype=torch.float32)
    else:
        embed_mean = torch.from_numpy(train_features.mean(axis=0).astype("float32"))
        embed_std = torch.from_numpy(np.clip(train_features.std(axis=0), a_min=1.0e-6, a_max=None).astype("float32"))

    model_kwargs = {
        "embed_dim": embed_dim,
        "out_dim": out_dim,
        "num_heads": num_heads,
        "use_bn_in_head": bool(_resolve_temi_param(params, hyperparameter_defaults, "use_bn_in_head", False)),
        "head_dropout_prob": float(
            _resolve_temi_param(params, hyperparameter_defaults, "head_dropout_prob", 0.0)
        ),
        "head_final_gelu": bool(_resolve_temi_param(params, hyperparameter_defaults, "head_final_gelu", False)),
        "norm_last_layer": bool(_resolve_temi_param(params, hyperparameter_defaults, "norm_last_layer", False)),
        "nlayers": int(_resolve_temi_param(params, hyperparameter_defaults, "nlayers", 2)),
        "hidden_dim": int(_resolve_temi_param(params, hyperparameter_defaults, "hidden_dim", 512)),
        "bottleneck_dim": int(_resolve_temi_param(params, hyperparameter_defaults, "bottleneck_dim", 256)),
        "l2_norm": bool(_resolve_temi_param(params, hyperparameter_defaults, "l2_norm", False)),
        "embed_norm": embed_norm,
        "embed_mean": embed_mean,
        "embed_std": embed_std,
    }
    student = TEMIHeadModel(**model_kwargs).to(device)
    teacher = TEMIHeadModel(**model_kwargs).to(device)
    student_teacher = TEMIStudentTeacher(student=student, teacher=teacher).to(device)

    if checkpoint_payload is not None:
        student_teacher.student.load_state_dict(checkpoint_payload["student"], strict=True)
        student_teacher.teacher.load_state_dict(checkpoint_payload["teacher"], strict=True)

    warmup_teacher_temp = float(_resolve_temi_param(params, hyperparameter_defaults, "warmup_teacher_temp", 0.1))
    teacher_temp = float(_resolve_temi_param(params, hyperparameter_defaults, "teacher_temp", 0.1))
    warmup_teacher_temp_epochs = int(
        _resolve_temi_param(params, hyperparameter_defaults, "warmup_teacher_temp_epochs", 30)
    )
    student_temp = float(_resolve_temi_param(params, hyperparameter_defaults, "student_temp", 0.1))
    beta = float(_resolve_temi_param(params, hyperparameter_defaults, "beta", 0.6))
    probs_momentum = float(_resolve_temi_param(params, hyperparameter_defaults, "probs_momentum", 0.9))
    positive_pmi = bool(_resolve_temi_param(params, hyperparameter_defaults, "positive_pmi", False))

    optimizer_name = str(_resolve_temi_param(params, hyperparameter_defaults, "optimizer", "adamw"))
    learning_rate = float(_resolve_temi_param(params, hyperparameter_defaults, "learning_rate", 1.0e-4, alias="lr"))
    min_learning_rate = float(
        _resolve_temi_param(params, hyperparameter_defaults, "min_learning_rate", 1.0e-4, alias="min_lr")
    )
    weight_decay = float(_resolve_temi_param(params, hyperparameter_defaults, "weight_decay", 1.0e-4))
    weight_decay_end = float(_resolve_temi_param(params, hyperparameter_defaults, "weight_decay_end", 1.0e-4))
    warmup_epochs = int(_resolve_temi_param(params, hyperparameter_defaults, "warmup_epochs", 20))
    momentum_teacher = float(_resolve_temi_param(params, hyperparameter_defaults, "momentum_teacher", 0.996))
    max_momentum_teacher = float(_resolve_temi_param(params, hyperparameter_defaults, "max_momentum_teacher", 0.996))
    clip_grad = float(_resolve_temi_param(params, hyperparameter_defaults, "clip_grad", 0.0))
    freeze_last_layer = int(_resolve_temi_param(params, hyperparameter_defaults, "freeze_last_layer", 1))
    use_fp16 = bool(_resolve_temi_param(params, hyperparameter_defaults, "use_fp16", False))

    training_history: list[dict[str, Any]] = []
    efficiency_measurement = None
    if not checkpoint_loaded:
        efficiency_measurement = start_train_eval_measurement(device)
        criterion = TEMILoss(
            out_dim=out_dim,
            batch_size=batch_size,
            num_heads=num_heads,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            epochs=epochs,
            student_temp=student_temp,
            beta=beta,
            probs_momentum=probs_momentum,
            positive_pmi=positive_pmi,
        ).to(device)
        parameter_groups = _get_params_groups(student_teacher.student.head)
        optimizer = _build_optimizer(optimizer_name, parameter_groups)
        niter_per_epoch = len(train_loader)
        lr_schedule = _cosine_scheduler(
            learning_rate * batch_size / 256.0,
            min_learning_rate * batch_size / 256.0,
            epochs,
            niter_per_epoch,
            warmup_epochs=warmup_epochs,
        )
        wd_schedule = _cosine_scheduler(weight_decay, weight_decay_end, epochs, niter_per_epoch)
        momentum_schedule = _cosine_scheduler(momentum_teacher, max_momentum_teacher, epochs, niter_per_epoch)
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and device.type == "cuda")
        progress.log(f"Training TEMI student/teacher heads for {epochs} epoch(s)")
        for epoch_index in range(epochs):
            epoch_stats = _train_one_epoch(
                student_teacher=student_teacher,
                criterion=criterion,
                dataloader=train_loader,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                wd_schedule=wd_schedule,
                momentum_schedule=momentum_schedule,
                epoch=epoch_index,
                scaler=scaler,
                use_fp16=use_fp16 and device.type == "cuda",
                device=device,
                clip_grad=clip_grad,
                freeze_last_layer=freeze_last_layer,
            )
            training_history.append(epoch_stats)
            progress.epoch(
                "TEMI training",
                epoch_index + 1,
                epochs,
                metrics={"loss": float(epoch_stats.get("loss", 0.0))},
            )

            if save_checkpoint_frequency > 0 and (
                (epoch_index + 1) % save_checkpoint_frequency == 0 or epoch_index + 1 == epochs
            ):
                torch.save(
                    {
                        "epoch": epoch_index + 1,
                        "student": student_teacher.student.state_dict(),
                        "teacher": student_teacher.teacher.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "criterion": criterion.state_dict(),
                        "metadata": {
                            "out_dim": out_dim,
                            "num_heads": num_heads,
                            "embed_dim": embed_dim,
                        },
                    },
                    checkpoint_dir / f"temi_epoch_{epoch_index + 1}.pt",
                )

    with progress.stage(f"Predicting evaluation split '{eval_dataset.name}/{eval_split}' with TEMI teacher"):
        prediction_array = _predict_with_teacher(
            student_teacher.teacher,
            eval_features,
            batch_size=int(params.get("eval_batch_size", batch_size)),
            device=device,
        )
    if efficiency_measurement is not None:
        efficiency_measurement.stop()

    return TEMIOutputs(
        predictions=prediction_array.tolist(),
        evaluation_labels=None if eval_labels is None else eval_labels.astype(np.int64).tolist(),
        evaluation_split=eval_split,
        metadata={
            "source_dataset": train_dataset.name,
            "evaluation_dataset": eval_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": eval_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "dataset_hyperparameter_profile": _normalize_dataset_name(train_dataset.name)
            if hyperparameter_defaults
            else "default",
            "backbone_hyperparameter_profile": backbone_hyperparameter_profile,
            "out_dim": out_dim,
            "out_dim_source": out_dim_source,
            "knn": knn,
            "num_heads": num_heads,
            "best_head_idx": int(student_teacher.teacher.head.best_head_idx.item()),
            "image_batch_size": image_batch_size,
            "batch_size": batch_size,
            "requested_batch_size": requested_batch_size,
            "eval_batch_size": int(params.get("eval_batch_size", batch_size)),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "weight_decay": weight_decay,
            "weight_decay_end": weight_decay_end,
            "warmup_epochs": warmup_epochs,
            "momentum_teacher": momentum_teacher,
            "max_momentum_teacher": max_momentum_teacher,
            "warmup_teacher_temp": warmup_teacher_temp,
            "teacher_temp": teacher_temp,
            "warmup_teacher_temp_epochs": warmup_teacher_temp_epochs,
            "student_temp": student_temp,
            "beta": beta,
            "probs_momentum": probs_momentum,
            "positive_pmi": positive_pmi,
            "optimizer": optimizer_name,
            "clip_grad": clip_grad,
            "freeze_last_layer": freeze_last_layer,
            "use_fp16": use_fp16,
            "embed_aug": embed_aug,
            "num_augs": num_augs,
            "gaussian_std": gaussian_std,
            "l2_norm": bool(_resolve_temi_param(params, hyperparameter_defaults, "l2_norm", False)),
            "embed_norm": embed_norm,
            "nlayers": int(_resolve_temi_param(params, hyperparameter_defaults, "nlayers", 2)),
            "hidden_dim": int(_resolve_temi_param(params, hyperparameter_defaults, "hidden_dim", 512)),
            "bottleneck_dim": int(_resolve_temi_param(params, hyperparameter_defaults, "bottleneck_dim", 256)),
            "norm_last_layer": bool(_resolve_temi_param(params, hyperparameter_defaults, "norm_last_layer", False)),
            "artifact_dir": str(artifact_dir),
            "save_checkpoint": save_checkpoint,
            "save_checkpoint_frequency": save_checkpoint_frequency,
            "load_checkpoint": load_checkpoint,
            "checkpoint_loaded": checkpoint_loaded,
            "loaded_checkpoint_path": None if loaded_checkpoint_path is None else str(loaded_checkpoint_path),
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "train_features_loaded": train_features is not None,
            "train_feature_count": int(train_features.shape[0]) if train_features is not None else train_dataset.num_samples,
            "evaluation_feature_count": int(eval_features.shape[0]),
            "shared_feature_cache": "common raw OpenCLIP image features",
            "knn_graph_built": not checkpoint_loaded,
            "knn_cache": None if checkpoint_loaded else str(knn_cache_path),
            "training_history": training_history,
            "protocol_note": (
                "TEMI official code trains a student/teacher multi-head clustering head over precomputed embeddings "
                "and KNN pairs. This benchmark version keeps that objective and training flow, replaces the input "
                "embeddings with shared OpenCLIP image features, trains only on the source train split, and evaluates "
                "the EMA teacher best head once on the evaluation split without using evaluation labels for model "
                "selection."
            ),
        },
    )


def _train_one_epoch(
    *,
    student_teacher: TEMIStudentTeacher,
    criterion: TEMILoss,
    dataloader,
    optimizer,
    lr_schedule: np.ndarray,
    wd_schedule: np.ndarray,
    momentum_schedule: np.ndarray,
    epoch: int,
    scaler,
    use_fp16: bool,
    device,
    clip_grad: float,
    freeze_last_layer: int,
) -> dict[str, Any]:
    torch = require_module("torch", "pip install torch")
    student_teacher.train()
    criterion.train()
    epoch_loss = 0.0
    head_loss_sum = None
    batches = 0

    for batch_index, (views, _) in enumerate(dataloader):
        iteration = len(dataloader) * epoch + batch_index
        for group_index, parameter_group in enumerate(optimizer.param_groups):
            parameter_group["lr"] = float(lr_schedule[iteration])
            if group_index == 0:
                parameter_group["weight_decay"] = float(wd_schedule[iteration])

        views = [view.to(device=device, dtype=torch.float32, non_blocking=True) for view in views]
        with torch.cuda.amp.autocast(enabled=use_fp16):
            teacher_out, student_out = student_teacher(views)
            head_losses = criterion(student_out, teacher_out, epoch=epoch)
            loss = head_losses.mean()

        if not math.isfinite(float(loss.item())):
            raise FloatingPointError(f"TEMI loss is non-finite: {float(loss.item())}.")

        optimizer.zero_grad()
        if use_fp16:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                _clip_gradients(student_teacher.student.head, clip_grad)
            _cancel_gradients_last_layer(epoch, student_teacher, freeze_last_layer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                _clip_gradients(student_teacher.student.head, clip_grad)
            _cancel_gradients_last_layer(epoch, student_teacher, freeze_last_layer)
            optimizer.step()

        with torch.no_grad():
            momentum = float(momentum_schedule[iteration])
            for student_param, teacher_param in zip(
                student_teacher.student.head.parameters(),
                student_teacher.teacher.head.parameters(),
            ):
                teacher_param.data.mul_(momentum).add_((1.0 - momentum) * student_param.detach().data)

        epoch_loss += float(loss.item())
        detached_head_losses = head_losses.detach().cpu()
        head_loss_sum = detached_head_losses if head_loss_sum is None else head_loss_sum + detached_head_losses
        batches += 1

    if head_loss_sum is None:
        raise ValueError("TEMI training epoch received no batches.")
    average_head_losses = head_loss_sum / batches
    student_teacher.student.head.set_losses(average_head_losses.to(device))
    student_teacher.teacher.head.set_losses(average_head_losses.to(device))
    return {
        "epoch": epoch + 1,
        "loss": epoch_loss / max(batches, 1),
        "head_losses": [float(value) for value in average_head_losses.tolist()],
        "best_head_idx": int(student_teacher.teacher.head.best_head_idx.item()),
        "lr": float(lr_schedule[min((epoch + 1) * len(dataloader) - 1, len(lr_schedule) - 1)]),
        "weight_decay": float(wd_schedule[min((epoch + 1) * len(dataloader) - 1, len(wd_schedule) - 1)]),
    }


@require_module("torch", "pip install torch").no_grad()
def _predict_with_teacher(model: TEMIHeadModel, features: np.ndarray, batch_size: int, device) -> np.ndarray:
    torch = require_module("torch", "pip install torch")
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch = torch.from_numpy(features[start : start + batch_size].astype("float32")).to(device)
        logits = model(batch)
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64))
    return np.concatenate(predictions, axis=0)


def _load_or_compute_train_neighbors(
    features: np.ndarray,
    *,
    k: int,
    cache_path: Path,
    device,
    chunk_size: int,
    force_recompute: bool,
) -> np.ndarray:
    if cache_path.exists() and not force_recompute:
        return np.load(cache_path)["neighbors"]

    torch = require_module("torch", "pip install torch")
    normalized = torch.from_numpy(features.astype("float32"))
    normalized = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    normalized = normalized.to(device)
    neighbors = []
    chunk_size = max(chunk_size, 1)
    with torch.no_grad():
        for start in range(0, normalized.size(0), chunk_size):
            end = min(start + chunk_size, normalized.size(0))
            similarities = normalized[start:end] @ normalized.T
            row = torch.arange(end - start, device=device)
            col = torch.arange(start, end, device=device)
            similarities[row, col] = -torch.inf
            indices = similarities.topk(k, dim=1).indices.cpu().numpy().astype(np.int64)
            neighbors.append(indices)
    neighbor_array = np.concatenate(neighbors, axis=0)
    np.savez_compressed(cache_path, neighbors=neighbor_array)
    return neighbor_array


def _sim_weight(first, second, gamma: float = 1.0):
    return (first * second).pow(gamma).sum(dim=-1)


def _beta_mi(first, second, pk, *, beta: float, clip_min):
    beta_emi = (((first * second) ** beta) / pk).sum(dim=-1)
    return -beta_emi.log().clamp(min=clip_min)


def _update_ema_probs(outputs, ema, momentum: float):
    if isinstance(outputs, tuple):
        outputs = require_module("torch", "pip install torch").cat(outputs)
    else:
        outputs = require_module("torch", "pip install torch").cat(list(outputs))
    batch_center = outputs.sum(dim=0, keepdim=True) / len(outputs)
    return ema * momentum + batch_center.squeeze(0) * (1.0 - momentum)


def _build_optimizer(name: str, parameter_groups):
    torch = require_module("torch", "pip install torch")
    normalized = name.strip().lower()
    if normalized == "adamw":
        return torch.optim.AdamW(parameter_groups)
    if normalized == "sgd":
        return torch.optim.SGD(parameter_groups, lr=0, momentum=0.9)
    raise ValueError(f"TEMI supports optimizer='adamw' or 'sgd', got '{name}'.")


def _normalize_temi_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _uses_temi_generic_openclip_defaults(
    openclip_pretraining: str | None,
    _openclip_backbone: str | None,
) -> bool:
    if openclip_pretraining is None:
        return False
    return _normalize_temi_profile_token(openclip_pretraining) == "siglip"


def _resolve_temi_checkpoint_artifact_dir(
    artifact_dir: Path,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> Path:
    if _uses_temi_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return artifact_dir / TEMI_GENERIC_OPENCLIP_CACHE_TOKEN
    return artifact_dir


def _resolve_temi_backbone_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> dict[str, Any]:
    if _uses_temi_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return {}
    profile_key = (
        _normalize_temi_profile_token(openclip_pretraining),
        _normalize_temi_profile_token(openclip_backbone),
        _normalize_dataset_name(dataset_name),
    )
    return dict(TEMI_BACKBONE_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_temi_backbone_hyperparameter_profile_source(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> str | None:
    if _uses_temi_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return None
    profile_key = (
        _normalize_temi_profile_token(openclip_pretraining),
        _normalize_temi_profile_token(openclip_backbone),
        _normalize_dataset_name(dataset_name),
    )
    if profile_key not in TEMI_BACKBONE_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_temi_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
    openclip_backbone: str | None = None,
) -> dict[str, Any]:
    if _uses_temi_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return {}
    defaults = dict(TEMI_DATASET_HYPERPARAMETER_DEFAULTS.get(_normalize_dataset_name(dataset_name), {}))
    if openclip_pretraining is not None and openclip_backbone is not None:
        defaults.update(
            _resolve_temi_backbone_hyperparameter_defaults(
                dataset_name,
                openclip_pretraining=openclip_pretraining,
                openclip_backbone=openclip_backbone,
            )
        )
    return defaults


def _resolve_temi_param(
    params: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
    fallback: Any,
    *,
    alias: str | None = None,
) -> Any:
    if key in params and params[key] is not None:
        return params[key]
    if alias is not None and alias in params and params[alias] is not None:
        return params[alias]
    return defaults.get(key, fallback)


def _validate_temi_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    out_dim: int,
    num_heads: int,
    checkpoint_path: Path,
) -> None:
    expected_values = {"out_dim": out_dim, "num_heads": num_heads}
    for key, expected in expected_values.items():
        if key not in metadata:
            continue
        actual = int(metadata[key])
        if actual != expected:
            raise ValueError(
                f"TEMI checkpoint architecture mismatch for {key}: checkpoint has {actual}, "
                f"current run expects {expected}. Checkpoint: {checkpoint_path}"
            )


def _get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or len(parameter.shape) == 1:
            not_regularized.append(parameter)
        else:
            regularized.append(parameter)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def _cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_epoch: int,
    *,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    warmup_iters = warmup_epochs * niter_per_epoch
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    remaining_iters = max(epochs * niter_per_epoch - warmup_iters, 0)
    if remaining_iters == 0:
        schedule = warmup_schedule
    else:
        iters = np.arange(remaining_iters)
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        schedule = np.concatenate((warmup_schedule, cosine_schedule))
    if len(schedule) < epochs * niter_per_epoch:
        schedule = np.pad(schedule, (0, epochs * niter_per_epoch - len(schedule)), constant_values=final_value)
    return schedule[: epochs * niter_per_epoch]


def _clip_gradients(model, clip: float):
    torch = require_module("torch", "pip install torch")
    norms = []
    for _, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.data.norm(2)
        norms.append(parameter_norm)
        clip_coef = clip / (parameter_norm + 1.0e-6)
        if clip_coef < 1:
            parameter.grad.data.mul_(clip_coef)
    return torch.stack(norms) if norms else torch.tensor([])


def _cancel_gradients_last_layer(epoch: int, model: TEMIStudentTeacher, freeze_last_layer: int) -> None:
    if epoch >= freeze_last_layer:
        return
    for name, parameter in model.student.named_parameters():
        if "last_layer" in name:
            parameter.grad = None


def _artifact_token(
    *,
    source_dataset: str,
    train_split: str,
    evaluation_dataset: str,
    evaluation_split: str,
    out_dim: int,
    knn: int,
) -> str:
    source_token = _normalize_dataset_name(source_dataset)
    eval_token = _normalize_dataset_name(evaluation_dataset)
    return f"{source_token}__{train_split}__to__{eval_token}__{evaluation_split}__out{out_dim}__knn{knn}"
