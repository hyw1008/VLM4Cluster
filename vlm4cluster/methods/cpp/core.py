from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.evaluation.metrics import evaluate_clustering
from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_dir as ensure_checkpoint_dir
from vlm4cluster.methods.checkpointing import checkpoint_interval, save_checkpoint_enabled
from vlm4cluster.methods.checkpointing import first_existing_checkpoint, latest_checkpoint
from vlm4cluster.methods.checkpointing import load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.feature_cache import (
    load_or_compute_raw_image_embeddings,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
)
from vlm4cluster.methods.tac.core import _normalize_dataset_name, _resolve_domain_shift_datasets
from vlm4cluster.models import load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import run_faiss_kmeans
from vlm4cluster.utils.io import ensure_dir
from vlm4cluster.utils.progress import get_progress_logger


CPP_OFFICIAL_DEFAULTS: dict[str, Any] = {
    "hidden_dim": 4096,
    "z_dim": 128,
    "epochs": 30,
    "batch_size": 2000,
    "learning_rate": 3.0e-3,
    "cluster_learning_rate": 2.0e-3,
    "momentum": 0.9,
    "pi_regularization": 0.05,
    "weight_decay_main": 1.0e-4,
    "weight_decay_cluster": 5.0e-3,
    "eps": 0.1,
    "gamma": 1.0,
    "pieta": 0.175,
    "piiter": 5,
    "warmup_steps": 0,
}

CPP_EPOCH_EVALUATION_METRICS = ("nmi", "ari", "acc")

CPP_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

CPP_GENERIC_OPENCLIP_ARTIFACT_VERSION = "generic-openclip-defaults-v1"

CPP_DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "cifar10": {
        "epochs": 5,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.175,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 50,
    },
    "cifar20": {
        "epochs": 15,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.13,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 50,
    },
    "cifar100": {
        "epochs": 50,
        "batch_size": 1500,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 33,
    },
    "stl10": {
        "epochs": 15,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.175,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 5,
    },
    "imagenet10": {
        "epochs": 15,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.175,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 50,
    },
    "imagenetdogs": {
        "epochs": 15,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.13,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 50,
    },
    "imagenet": {
        "epochs": 20,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.12,
        "hidden_dim": 2048,
        "z_dim": 1024,
        "warmup_steps": 2000,
    },
    "places365standard": {
        "epochs": 20,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 2048,
        "z_dim": 512,
        "warmup_steps": 1000,
    },
    "dtd": {
        "epochs": 30,
        "batch_size": 512,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 10,
    },
    "aircraft": {
        "epochs": 50,
        "batch_size": 512,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 10,
    },
    "cars": {
        "epochs": 50,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 256,
        "warmup_steps": 10,
    },
    "flowers": {
        "epochs": 50,
        "batch_size": 512,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 10,
    },
    "food": {
        "epochs": 30,
        "batch_size": 1500,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 50,
    },
    "pets": {
        "epochs": 30,
        "batch_size": 512,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 10,
    },
    "ucf101": {
        "epochs": 30,
        "batch_size": 1024,
        "learning_rate": 1.0e-4,
        "cluster_learning_rate": 1.0e-4,
        "pieta": 0.1,
        "hidden_dim": 4096,
        "z_dim": 128,
        "warmup_steps": 100,
    },
}

CPP_BACKBONE_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10"): {"head_mode": "single_head"},
    ("laion400m", "vitb32", "cifar100"): {"head_mode": "single_head"},
    ("laion400m", "vitb32", "imagenetdogs"): {"pieta": 0.2},
    ("laion400m", "vitb32", "dtd"): {"pieta": 0.225},
    ("laion400m", "vitb32", "ucf101"): {"pieta": 0.2},
    ("laion400m", "vitb32", "imagenet"): {"head_mode": "single_head"},
    ("laion400m", "vitb32", "places365standard"): {"head_mode": "single_head", "pieta": 0.2},
    ("laion400m", "vitb32", "aircraft"): {"pieta": 0.2},
    ("laion400m", "vitb16", "cifar20"): {"head_mode": "single_head"},
    ("laion400m", "vitb16", "imagenet10"): {"head_mode": "single_head"},
    ("laion400m", "vitb16", "imagenetdogs"): {"pieta": 0.2},
    ("laion400m", "vitb16", "dtd"): {"pieta": 0.2},
    ("laion400m", "vitb16", "ucf101"): {"pieta": 0.2},
    ("laion400m", "vitb16", "imagenet"): {"head_mode": "single_head"},
    ("laion400m", "vitb16", "places365standard"): {"head_mode": "single_head", "pieta": 0.2},
    ("laion400m", "vitb16", "aircraft"): {"pieta": 0.2},
    ("laion400m", "vitb16", "pets"): {"pieta": 0.15},
    ("laion400m", "vitl14", "cifar10"): {"head_mode": "single_head"},
    ("laion400m", "vitl14", "cifar100"): {"head_mode": "single_head"},
    ("laion400m", "vitl14", "imagenetdogs"): {"pieta": 0.2},
    ("laion400m", "vitl14", "dtd"): {"pieta": 0.15},
    ("laion400m", "vitl14", "ucf101"): {"pieta": 0.2},
    ("laion400m", "vitl14", "imagenet"): {"head_mode": "single_head"},
    ("laion400m", "vitl14", "places365standard"): {"head_mode": "single_head", "pieta": 0.2},
    ("laion400m", "vitl14", "aircraft"): {"pieta": 0.2},
    ("laion400m", "vitl14", "pets"): {"pieta": 0.15},
}


@dataclass(slots=True)
class CPPOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class CPPNetwork(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.pre_feature = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.subspace = torch.nn.Sequential(torch.nn.Linear(hidden_dim, z_dim))
        self.cluster = torch.nn.Sequential(torch.nn.Linear(hidden_dim, z_dim))

    def forward(self, x):
        torch = require_module("torch", "pip install torch")
        pre_feature = self.pre_feature(x)
        z = self.subspace(pre_feature)
        logits = self.cluster(pre_feature)
        z = torch.nn.functional.normalize(z, dim=1)
        logits = torch.nn.functional.normalize(logits, dim=1)
        return z, logits


class MLCLoss(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, eps: float = 0.01, gamma: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discriminative_loss(self, weights):
        torch = require_module("torch", "pip install torch")
        rows, cols = weights.shape
        identity = torch.eye(rows, device=weights.device, dtype=weights.dtype)
        scalar = rows / (cols * self.eps)
        logdet = torch.logdet(identity + scalar * weights.matmul(weights.T))
        return logdet / 2.0

    def compute_compress_loss(self, weights, membership):
        torch = require_module("torch", "pip install torch")
        rows, cols = weights.shape
        clusters, _, _ = membership.shape
        identity = torch.eye(rows, device=weights.device, dtype=weights.dtype).expand((clusters, rows, rows))
        trace_membership = membership.sum(2) + 1.0e-8
        scale = (rows / (trace_membership * self.eps)).view(clusters, 1, 1)
        reshaped_weights = weights.view((1, rows, cols))
        logdet = torch.logdet(
            identity + scale * reshaped_weights.mul(membership).matmul(reshaped_weights.transpose(1, 2))
        )
        return (trace_membership.squeeze() * logdet / (2 * cols)).sum()

    def forward(self, z, membership_matrix):
        membership = membership_matrix.T.reshape((membership_matrix.shape[0], 1, -1))
        weights = z.T
        discriminative_loss = self.compute_discriminative_loss(weights)
        compress_loss = self.compute_compress_loss(weights, membership)
        total_loss = -discriminative_loss + self.gamma * compress_loss
        return total_loss, discriminative_loss, compress_loss


class TotalCodingRate(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, eps: float = 0.01) -> None:
        super().__init__()
        self.eps = eps

    def compute_discriminative_loss(self, weights):
        torch = require_module("torch", "pip install torch")
        rows, cols = weights.shape
        identity = torch.eye(rows, device=weights.device, dtype=weights.dtype)
        scalar = rows / (cols * self.eps)
        logdet = torch.logdet(identity + scalar * weights.matmul(weights.T))
        return logdet / 2.0

    def forward(self, x):
        return -self.compute_discriminative_loss(x.T)


class SinkhornDistance(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, eps: float, max_iter: int) -> None:
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, cost):
        torch = require_module("torch", "pip install torch")
        matrix = -cost
        x_points = matrix.shape[-2]
        y_points = matrix.shape[-1]
        batch_size = matrix.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False, device=matrix.device).fill_(
            1.0 / x_points
        )
        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False, device=matrix.device).fill_(
            1.0 / y_points
        )
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        threshold = 1.0e-12
        error = None

        for index in range(self.max_iter):
            if index % 2 == 0:
                previous_u = u
                u = self.eps * (torch.log(mu) - torch.logsumexp(self._m(matrix, u, v), dim=-1)) + u
                error = (u - previous_u).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self._m(matrix, u, v).transpose(-2, -1), dim=-1)) + v
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1.0e8] = 0.0
                v = v.detach().requires_grad_(True)

            if error is not None and float(error.item()) < threshold:
                break

        pi = torch.exp(self._m(matrix, u, v))
        return pi, matrix, u, v

    def _m(self, cost, u, v):
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


def run_cpp_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> CPPOutputs:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("cpp", params)

    seed = int(params.get("seed", 42))
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
        "cpp",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="CPP",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if eval_dataset.labels is None:
        raise ValueError("CPP requires labels on the evaluation split.")

    eval_cluster_num = int(params.get("eval_n_clusters", params.get("n_clusters", eval_dataset.num_classes)))
    if eval_cluster_num <= 0:
        raise ValueError("CPP requires n_clusters/eval_n_clusters > 0.")

    dataset_profile = _resolve_cpp_dataset_hyperparameter_defaults(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    hyperparameter_artifact_version = _resolve_cpp_hyperparameter_artifact_version(
        openclip_pretraining,
        openclip_backbone,
    )
    hyperparameter_defaults = _resolve_cpp_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    hyperparameter_profile_source = _resolve_cpp_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    profile = {**dataset_profile, **hyperparameter_defaults}
    image_batch_size = int(params.get("image_batch_size", 256))
    with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
        train_features, _ = load_or_compute_raw_image_embeddings(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )
    with progress.stage(f"Encoding evaluation image split '{eval_dataset.name}/{eval_split}'"):
        eval_features, eval_labels = load_or_compute_raw_image_embeddings(
            dataset=eval_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )

    input_dim = int(train_features.shape[1])
    hidden_dim = int(_resolve_param(params, "hidden_dim", default=_profile_default(profile, "hidden_dim")))
    z_dim = int(_resolve_param(params, "z_dim", default=_profile_default(profile, "z_dim")))
    epochs = int(_resolve_param(params, "epochs", alias="epo", default=_profile_default(profile, "epochs")))
    batch_size = int(_resolve_param(params, "batch_size", alias="bs", default=_profile_default(profile, "batch_size")))
    learning_rate = float(
        _resolve_param(params, "learning_rate", alias="lr", default=_profile_default(profile, "learning_rate"))
    )
    cluster_learning_rate = float(
        _resolve_param(
            params,
            "cluster_learning_rate",
            alias="lr_c",
            default=_profile_default(profile, "cluster_learning_rate"),
        )
    )
    momentum = float(_resolve_param(params, "momentum", alias="momo", default=_profile_default(profile, "momentum")))
    pi_regularization = float(
        _resolve_param(params, "pi_regularization", alias="pigam", default=_profile_default(profile, "pi_regularization"))
    )
    weight_decay_main = float(
        _resolve_param(params, "weight_decay_main", alias="wd1", default=_profile_default(profile, "weight_decay_main"))
    )
    weight_decay_cluster = float(
        _resolve_param(
            params,
            "weight_decay_cluster",
            alias="wd2",
            default=_profile_default(profile, "weight_decay_cluster"),
        )
    )
    eps = float(_resolve_param(params, "eps", default=_profile_default(profile, "eps")))
    gamma = float(_resolve_param(params, "gamma", default=_profile_default(profile, "gamma")))
    pieta = float(_resolve_param(params, "pieta", default=_profile_default(profile, "pieta")))
    piiter = int(_resolve_param(params, "piiter", default=_profile_default(profile, "piiter")))
    warmup_steps = int(
        _resolve_param(params, "warmup_steps", alias="warmup", default=_profile_default(profile, "warmup_steps"))
    )
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)
    save_every = checkpoint_interval(params, enabled=save_checkpoint, default=50)
    spectral_kmeans_n_init = int(params.get("spectral_kmeans_n_init", 10))
    spectral_random_state = params.get("spectral_random_state", seed)
    spectral_solver_type = str(params.get("spectral_solver_type", "lm"))
    spectral_tol = float(params.get("spectral_tol", 0.0))
    normalize_spectral_embedding = bool(params.get("normalize_spectral_embedding", True))
    head_mode = _resolve_head_mode(params, hyperparameter_defaults)
    select_best_epoch = _as_bool(params.get("select_best_epoch", True))
    eval_every_epochs = int(params.get("eval_every_epochs", params.get("validate_every_epochs", 1)))
    best_epoch_metric = _resolve_best_epoch_metric(params)

    if batch_size <= 0:
        raise ValueError("CPP requires batch_size > 0.")
    if select_best_epoch and eval_every_epochs <= 0:
        raise ValueError("CPP requires eval_every_epochs > 0 when select_best_epoch=true.")
    if len(train_features) < 2:
        raise ValueError("CPP training requires at least two training samples.")
    effective_batch_size = min(batch_size, len(train_features))
    if effective_batch_size != batch_size:
        progress.log(
            f"Reducing CPP batch size from {batch_size} to {effective_batch_size} "
            f"for {len(train_features)} training sample(s)"
        )
    if eval_cluster_num >= len(eval_features):
        raise ValueError(
            f"CPP requires eval_n_clusters < number of evaluation samples, got {eval_cluster_num} and "
            f"{len(eval_features)} samples."
        )

    eval_features_float = eval_features.astype("float32", copy=False)
    train_tensor = torch.from_numpy(train_features.astype("float32"))
    train_loader = data_mod.DataLoader(
        data_mod.TensorDataset(train_tensor),
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError("CPP training loader is empty after applying drop_last=True.")

    model = CPPNetwork(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim).to(device)
    sink_layer = SinkhornDistance(pieta, max_iter=piiter)
    criterion = MLCLoss(eps=eps, gamma=gamma)
    warmup_criterion = TotalCodingRate(eps=eps)
    optimizer = torch.optim.SGD(
        list(model.pre_feature.parameters()) + list(model.subspace.parameters()),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay_main,
        nesterov=False,
    )
    optimizer_cluster = torch.optim.SGD(
        model.cluster.parameters(),
        lr=cluster_learning_rate,
        momentum=momentum,
        weight_decay=weight_decay_cluster,
        nesterov=False,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    artifact_dir = ensure_dir(
        cache_root
        / _artifact_token(
            source_dataset=train_dataset.name,
            train_split=train_split,
            evaluation_dataset=eval_dataset.name,
            evaluation_split=eval_split,
            eval_cluster_num=eval_cluster_num,
            head_mode=head_mode,
            hyperparameter_artifact_version=hyperparameter_artifact_version,
        )
    )
    checkpoint_dir = ensure_checkpoint_dir(artifact_dir) if save_checkpoint else None
    source_eval_cluster_num = train_dataset.num_classes or eval_cluster_num
    source_artifact_dir = cache_root / _artifact_token(
        source_dataset=train_dataset.name,
        train_split=train_split,
        evaluation_dataset=train_dataset.name,
        evaluation_split="val",
        eval_cluster_num=source_eval_cluster_num,
        head_mode=head_mode,
        hyperparameter_artifact_version=hyperparameter_artifact_version,
    )
    load_checkpoint = load_checkpoint_enabled(params, train_dataset.name)
    existing_checkpoint = (
        first_existing_checkpoint(
            [
                path
                for path in (
                    latest_checkpoint(artifact_dir / "checkpoints", "cpp_epoch_*.pt"),
                    latest_checkpoint(source_artifact_dir / "checkpoints", "cpp_epoch_*.pt"),
                )
                if path is not None
            ]
        )
        if load_checkpoint
        else None
    )
    checkpoint_loaded = False
    loaded_checkpoint_path = None
    if load_checkpoint and existing_checkpoint is not None:
        progress.log(f"Loading CPP checkpoint and skipping training: {existing_checkpoint}")
        payload = load_torch_checkpoint(existing_checkpoint, device=device)
        model.load_state_dict(payload["model"], strict=True)
        checkpoint_loaded = True
        loaded_checkpoint_path = existing_checkpoint

    warmup_step = 0
    cluster_head_bootstrapped = False
    training_history: list[dict[str, float | int | bool]] = []
    epoch_evaluation_history: list[dict[str, float | int | str]] = []
    eval_label_list = eval_labels.astype(np.int64).tolist()
    best_epoch_record: dict[str, float | int | str] | None = None
    best_epoch_predictions: np.ndarray | None = None
    best_epoch_score = float("-inf")
    efficiency_measurement = None
    if not checkpoint_loaded:
        efficiency_measurement = start_train_eval_measurement(device)
        progress.log(f"Training CPP for {epochs} epoch(s)")
    for epoch_index in range(0 if checkpoint_loaded else epochs):
        model.train()
        epoch_total = 0.0
        epoch_tcr = 0.0
        epoch_discriminative = 0.0
        epoch_compress = 0.0
        epoch_pi_regularization = 0.0
        epoch_batches = 0
        warmup_active = warmup_step <= warmup_steps

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z, logits = model(x_batch)
                membership_codes = _membership_codes(z, logits, head_mode=head_mode)
                membership_matrix = _build_membership_matrix(membership_codes, sink_layer)
                if warmup_step <= warmup_steps:
                    loss_tcr = warmup_criterion(z)
                    discriminative_loss = None
                    compress_loss = None
                    loss = loss_tcr
                    pi_reg_loss = torch.zeros((), device=device, dtype=z.dtype)
                else:
                    loss, discriminative_loss, compress_loss = criterion(z, membership_matrix)
                    pi_reg_loss = pi_regularization * 0.5 * membership_matrix.norm() ** 2
                    loss = loss + pi_reg_loss
                    loss_tcr = None

            optimizer.zero_grad()
            optimizer_cluster.zero_grad()
            scaler.scale(loss).backward()
            cluster_has_grad = _optimizer_has_grad(optimizer_cluster)
            scaler.step(optimizer)
            if cluster_has_grad:
                scaler.step(optimizer_cluster)
            scaler.update()

            if head_mode == "two_head" and warmup_step == warmup_steps and not cluster_head_bootstrapped:
                _update_cluster_from_subspace(model)
                cluster_head_bootstrapped = True

            epoch_total += float(loss.item())
            if loss_tcr is not None:
                epoch_tcr += float(loss_tcr.item())
            if discriminative_loss is not None:
                epoch_discriminative += float(discriminative_loss.item())
            if compress_loss is not None:
                epoch_compress += float(compress_loss.item())
            epoch_pi_regularization += float(pi_reg_loss.item())
            epoch_batches += 1
            warmup_step += 1

        epoch_record = {
            "epoch": epoch_index + 1,
            "warmup_active": warmup_active,
            "loss_total": epoch_total / max(epoch_batches, 1),
            "loss_tcr": epoch_tcr / max(epoch_batches, 1) if warmup_active else 0.0,
            "loss_discriminative": epoch_discriminative / max(epoch_batches, 1) if not warmup_active else 0.0,
            "loss_compress": epoch_compress / max(epoch_batches, 1) if not warmup_active else 0.0,
            "loss_pi_regularization": epoch_pi_regularization / max(epoch_batches, 1),
        }

        should_evaluate_epoch = select_best_epoch and (
            (epoch_index + 1) % eval_every_epochs == 0 or epoch_index + 1 == epochs
        )
        if should_evaluate_epoch:
            with progress.stage(
                f"Evaluating CPP on '{eval_dataset.name}/{eval_split}' after epoch {epoch_index + 1}"
            ):
                epoch_predictions = _predict_with_membership_spectral_clustering(
                    model=model,
                    sink_layer=sink_layer,
                    features=eval_features_float,
                    batch_size=batch_size,
                    device=device,
                    n_clusters=eval_cluster_num,
                    kmeans_n_init=spectral_kmeans_n_init,
                    random_state=None if spectral_random_state is None else int(spectral_random_state),
                    solver_type=spectral_solver_type,
                    tol=spectral_tol,
                    normalize_embedding=normalize_spectral_embedding,
                    head_mode=head_mode,
                )
            epoch_metrics = evaluate_clustering(
                eval_label_list,
                epoch_predictions.astype(np.int64, copy=False).tolist(),
                list(CPP_EPOCH_EVALUATION_METRICS),
            )
            epoch_eval_record: dict[str, float | int | str] = {
                "epoch": epoch_index + 1,
                "selection_metric": best_epoch_metric,
                **{f"test_{metric}": value for metric, value in epoch_metrics.items()},
            }
            epoch_evaluation_history.append(epoch_eval_record)
            epoch_record.update({f"test_{metric}": value for metric, value in epoch_metrics.items()})
            epoch_score = float(epoch_metrics[best_epoch_metric])
            if epoch_score > best_epoch_score:
                best_epoch_score = epoch_score
                best_epoch_predictions = epoch_predictions.copy()
                best_epoch_record = epoch_eval_record.copy()

        training_history.append(epoch_record)
        progress_metrics: dict[str, float | bool] = {
            "loss": float(epoch_record["loss_total"]),
            "warmup": warmup_active,
        }
        if should_evaluate_epoch:
            progress_metrics.update(
                {
                    f"test_{metric}": float(epoch_record[f"test_{metric}"])
                    for metric in CPP_EPOCH_EVALUATION_METRICS
                    if f"test_{metric}" in epoch_record
                }
            )
        progress.epoch("CPP training", epoch_index + 1, epochs, metrics=progress_metrics)

        if save_every > 0 and ((epoch_index + 1) % save_every == 0 or epoch_index + 1 == epochs):
            torch.save(
                {
                    "epoch": epoch_index + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_cluster": optimizer_cluster.state_dict(),
                },
                checkpoint_dir / f"cpp_epoch_{epoch_index + 1}.pt",
            )

    if best_epoch_predictions is not None:
        predictions = best_epoch_predictions
        progress.log(
            f"Selected CPP epoch {best_epoch_record['epoch']} by test {best_epoch_metric}="
            f"{best_epoch_score:.6f}"
        )
    else:
        with progress.stage(f"Running CPP membership spectral clustering with {eval_cluster_num} cluster(s)"):
            predictions = _predict_with_membership_spectral_clustering(
                model=model,
                sink_layer=sink_layer,
                features=eval_features_float,
                batch_size=batch_size,
                device=device,
                n_clusters=eval_cluster_num,
                kmeans_n_init=spectral_kmeans_n_init,
                random_state=None if spectral_random_state is None else int(spectral_random_state),
                solver_type=spectral_solver_type,
                tol=spectral_tol,
                normalize_embedding=normalize_spectral_embedding,
                head_mode=head_mode,
            )
    if efficiency_measurement is not None:
        efficiency_measurement.stop()
    return CPPOutputs(
        predictions=predictions.tolist(),
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
            "hyperparameter_default_profile": hyperparameter_profile_source,
            **(
                {"hyperparameter_artifact_version": hyperparameter_artifact_version}
                if hyperparameter_artifact_version is not None
                else {}
            ),
            "head_mode": head_mode,
            "membership_source": "cluster_logits" if head_mode == "two_head" else "subspace_z",
            "official_efficient_shortcut_used": head_mode == "single_head",
            "cluster_head_trained": head_mode == "two_head",
            "n_clusters": eval_cluster_num,
            "image_batch_size": image_batch_size,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "z_dim": z_dim,
            "epochs": epochs,
            "batch_size": effective_batch_size,
            "requested_batch_size": batch_size,
            "learning_rate": learning_rate,
            "cluster_learning_rate": cluster_learning_rate,
            "momentum": momentum,
            "pi_regularization": pi_regularization,
            "weight_decay_main": weight_decay_main,
            "weight_decay_cluster": weight_decay_cluster,
            "eps": eps,
            "gamma": gamma,
            "pieta": pieta,
            "piiter": piiter,
            "warmup_steps": warmup_steps,
            "save_checkpoint": save_checkpoint,
            "save_every": save_every,
            "load_checkpoint": load_checkpoint,
            "checkpoint_loaded": checkpoint_loaded,
            "loaded_checkpoint_path": None if loaded_checkpoint_path is None else str(loaded_checkpoint_path),
            "select_best_epoch": select_best_epoch,
            "eval_every_epochs": eval_every_epochs,
            "best_epoch_metric": best_epoch_metric,
            "best_epoch": None if best_epoch_record is None else int(best_epoch_record["epoch"]),
            "best_epoch_score": None if best_epoch_record is None else float(best_epoch_record[f"test_{best_epoch_metric}"]),
            "best_epoch_metrics": best_epoch_record,
            "spectral_kmeans_n_init": spectral_kmeans_n_init,
            "spectral_random_state": spectral_random_state,
            "spectral_solver_type": spectral_solver_type,
            "spectral_tol": spectral_tol,
            "normalize_spectral_embedding": normalize_spectral_embedding,
            "artifact_dir": str(artifact_dir),
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "train_feature_count": int(train_features.shape[0]),
            "evaluation_feature_count": int(eval_features.shape[0]),
            "shared_feature_cache": "common raw OpenCLIP image features",
            "training_history": training_history,
            "epoch_evaluation_history": epoch_evaluation_history,
            "protocol_note": (
                "CPP is trained on shared OpenCLIP image features, but its method logic follows the paper and "
                "the official two-head raw-image implementation by default. When head_mode='single_head', "
                "the official main_efficient.py shortcut `logits = z` is used so Pi is built directly from Z. "
                "By default, CPP evaluates the held-out evaluation split after each epoch and returns the "
                "predictions from the epoch with the best selected test metric."
            ),
        },
    )


def _build_membership_matrix(logits, sink_layer: SinkhornDistance):
    self_coeff = (logits @ logits.T).abs().unsqueeze(0)
    membership = sink_layer(self_coeff)[0]
    membership = membership * membership.shape[-1]
    return membership[0]


def _membership_codes(z, logits, *, head_mode: str):
    if head_mode == "single_head":
        return z
    if head_mode == "two_head":
        return logits
    raise ValueError(f"Unknown CPP head_mode '{head_mode}'.")


def _predict_with_membership_spectral_clustering(
    *,
    model: CPPNetwork,
    sink_layer: SinkhornDistance,
    features: np.ndarray,
    batch_size: int,
    device,
    n_clusters: int,
    kmeans_n_init: int,
    random_state: int | None,
    solver_type: str,
    tol: float,
    normalize_embedding: bool,
    head_mode: str,
) -> np.ndarray:
    torch = require_module("torch", "pip install torch")

    model.eval()
    code_list = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device=device, dtype=torch.float32)
            z, logits = model(batch)
            code_list.append(_membership_codes(z, logits, head_mode=head_mode))
        all_codes = torch.cat(code_list, dim=0)
        membership = _build_membership_matrix(all_codes, sink_layer)
    return _spectral_cluster(
        membership.detach().cpu().numpy(),
        n_clusters,
        kmeans_n_init,
        random_state,
        solver_type,
        tol,
        normalize_embedding,
    )


def _spectral_cluster(
    affinity: np.ndarray,
    n_clusters: int,
    kmeans_n_init: int,
    random_state: int | None,
    solver_type: str,
    tol: float,
    normalize_embedding: bool,
) -> np.ndarray:
    scipy_sparse = require_module("scipy.sparse", "pip install scipy")

    laplacian = scipy_sparse.csgraph.laplacian(affinity, normed=True)
    if solver_type == "shift_invert":
        _, embedding = scipy_sparse.linalg.eigsh(laplacian, k=n_clusters, sigma=1.0e-6, which="LM", tol=tol)
    elif solver_type == "la":
        _, embedding = scipy_sparse.linalg.eigsh(-laplacian, k=n_clusters, sigma=None, which="LA", tol=tol)
    elif solver_type == "lm":
        _, embedding = scipy_sparse.linalg.eigsh(
            2 * scipy_sparse.identity(laplacian.shape[0]) - laplacian,
            k=n_clusters,
            sigma=None,
            which="LM",
            tol=tol,
        )
    else:
        raise ValueError(f"Unknown CPP spectral solver_type '{solver_type}'.")

    if normalize_embedding:
        embedding = embedding / np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), a_min=1.0e-12, a_max=None)
    labels, _ = run_faiss_kmeans(
        embedding.astype("float32"),
        n_clusters=n_clusters,
        n_iter=300,
        n_redo=kmeans_n_init,
        spherical=False,
        random_state=random_state,
    )
    return labels.astype(np.int64, copy=False)


def _update_cluster_from_subspace(model: CPPNetwork) -> CPPNetwork:
    import copy

    state_dict = model.state_dict()
    updated = copy.deepcopy(state_dict)
    keys_to_rename = [key for key in updated if "subspace" in key]
    for key in keys_to_rename:
        prefix, suffix = key.split("subspace")
        updated[prefix + "cluster" + suffix] = updated.pop(key)
    state_dict.update(updated)
    model.load_state_dict(state_dict)
    return model


def _optimizer_has_grad(optimizer) -> bool:
    return any(param.grad is not None for group in optimizer.param_groups for param in group["params"])


def _resolve_head_mode(params: dict[str, Any], defaults: dict[str, Any] | None = None) -> str:
    if "single_head" in params:
        single_head = _as_bool(params["single_head"])
        return "single_head" if single_head else "two_head"
    if "use_single_head" in params:
        single_head = _as_bool(params["use_single_head"])
        return "single_head" if single_head else "two_head"

    defaults = defaults or {}
    raw_mode = params.get(
        "head_mode",
        params.get("membership_head", params.get("head", defaults.get("head_mode", "two_head"))),
    )
    token = str(raw_mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "double": "two_head",
        "double_head": "two_head",
        "double_heads": "two_head",
        "doublehead": "two_head",
        "two_head": "two_head",
        "two_heads": "two_head",
        "twohead": "two_head",
        "cluster_logits": "two_head",
        "cluster_head": "two_head",
        "main": "two_head",
        "single_head": "single_head",
        "single": "single_head",
        "singlehead": "single_head",
        "subspace": "single_head",
        "subspace_z": "single_head",
        "z": "single_head",
        "efficient": "single_head",
        "main_efficient": "single_head",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        supported = "two_head, single_head"
        raise ValueError(f"Unsupported CPP head_mode '{raw_mode}'. Supported values: {supported}.") from exc


def _resolve_best_epoch_metric(params: dict[str, Any]) -> str:
    metric = str(params.get("best_epoch_metric", params.get("selection_metric", "acc"))).strip().lower()
    if metric not in CPP_EPOCH_EVALUATION_METRICS:
        supported = ", ".join(CPP_EPOCH_EVALUATION_METRICS)
        raise ValueError(f"Unsupported CPP best_epoch_metric '{metric}'. Supported values: {supported}.")
    return metric


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _resolve_param(params: dict[str, Any], key: str, *, alias: str | None = None, default: Any) -> Any:
    if key in params:
        return params[key]
    if alias is not None and alias in params:
        return params[alias]
    return default


def _profile_default(profile: dict[str, Any], key: str) -> Any:
    return profile.get(key, CPP_OFFICIAL_DEFAULTS[key])


def _dataset_profile(dataset_name: str) -> dict[str, Any]:
    return {**CPP_OFFICIAL_DEFAULTS, **CPP_DATASET_PROFILES.get(_normalize_cpp_profile_dataset(dataset_name), {})}


def _uses_cpp_generic_openclip_defaults(openclip_pretraining: str | None) -> bool:
    if openclip_pretraining is None:
        return False
    return _normalize_cpp_profile_token(openclip_pretraining) == "siglip"


def _resolve_cpp_dataset_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
    openclip_backbone: str | None = None,
) -> dict[str, Any]:
    if _uses_cpp_generic_openclip_defaults(openclip_pretraining):
        return dict(CPP_OFFICIAL_DEFAULTS)
    return _dataset_profile(dataset_name)


def _resolve_cpp_hyperparameter_artifact_version(
    openclip_pretraining: str | None,
    openclip_backbone: str | None,
) -> str | None:
    if _uses_cpp_generic_openclip_defaults(openclip_pretraining):
        return CPP_GENERIC_OPENCLIP_ARTIFACT_VERSION
    return None


def _resolve_cpp_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    if _uses_cpp_generic_openclip_defaults(openclip_pretraining):
        return {}
    profile_key = (
        _normalize_cpp_profile_token(openclip_pretraining),
        _normalize_cpp_profile_token(openclip_backbone),
        _normalize_cpp_profile_dataset(dataset_name),
    )
    return dict(CPP_BACKBONE_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_cpp_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    if _uses_cpp_generic_openclip_defaults(openclip_pretraining):
        return None
    profile_key = (
        _normalize_cpp_profile_token(openclip_pretraining),
        _normalize_cpp_profile_token(openclip_backbone),
        _normalize_cpp_profile_dataset(dataset_name),
    )
    if profile_key not in CPP_BACKBONE_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _normalize_cpp_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_cpp_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in CPP_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _artifact_token(
    *,
    source_dataset: str,
    train_split: str,
    evaluation_dataset: str,
    evaluation_split: str,
    eval_cluster_num: int,
    head_mode: str,
    hyperparameter_artifact_version: str | None = None,
) -> str:
    source_token = _normalize_dataset_name(source_dataset)
    eval_token = _normalize_dataset_name(evaluation_dataset)
    token = f"{source_token}__{train_split}__to__{eval_token}__{evaluation_split}__k{eval_cluster_num}__{head_mode}"
    if hyperparameter_artifact_version is not None:
        token = f"{token}__{hyperparameter_artifact_version}"
    return token
