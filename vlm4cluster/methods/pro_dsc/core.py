from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


PRO_DSC_COMMON_OPTIMIZER_DEFAULTS: dict[str, Any] = {
    "hidden_dim": 4096,
    "learning_rate": 1.0e-4,
    "cluster_learning_rate": 1.0e-4,
    "momentum": 0.9,
    "weight_decay_main": 1.0e-4,
    "weight_decay_cluster": 5.0e-3,
    "eps": 0.1,
}

PRO_DSC_GENERIC_FALLBACK_VERSION = "generic-openclip-defaults-v1"


def _pro_dsc_profile(**overrides: Any) -> dict[str, Any]:
    return {**PRO_DSC_COMMON_OPTIMIZER_DEFAULTS, **overrides}


PRO_DSC_OFFICIAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "cifar10": _pro_dsc_profile(
        gamma=300,
        beta=600,
        pieta=0.175,
        piiter=1,
        z_dim=128,
        epochs=10,
        batch_size=1024,
        warmup_steps=200,
    ),
    "cifar20": _pro_dsc_profile(
        gamma=600,
        beta=300,
        pieta=0.13,
        piiter=1,
        z_dim=256,
        epochs=50,
        batch_size=1500,
        learning_rate=5.0e-5,
        warmup_steps=-1,
    ),
    "cifar100": _pro_dsc_profile(
        gamma=150,
        beta=500,
        pieta=0.1,
        piiter=1,
        z_dim=128,
        epochs=100,
        batch_size=1500,
        warmup_steps=200,
    ),
    "tinyimagenet": _pro_dsc_profile(
        gamma=200,
        beta=400,
        pieta=0.1,
        piiter=1,
        z_dim=256,
        epochs=100,
        batch_size=1500,
        warmup_steps=-1,
    ),
    "imagenet": _pro_dsc_profile(
        gamma=800,
        beta=600,
        pieta=0.09,
        piiter=1,
        z_dim=1024,
        epochs=100,
        batch_size=2048,
        warmup_steps=2000,
    ),
    "imagenetdogs": _pro_dsc_profile(
        gamma=300,
        beta=400,
        pieta=0.1,
        piiter=5,
        z_dim=128,
        epochs=200,
        batch_size=1024,
        warmup_steps=-1,
    ),
}

PRO_DSC_BENCHMARK_DEFAULTS: dict[str, dict[str, Any]] = {
    "stl10": _pro_dsc_profile(
        gamma=300,
        beta=600,
        pieta=0.175,
        piiter=1,
        z_dim=128,
        epochs=100,
        batch_size=1024,
        warmup_steps=200,
    ),
    "imagenet10": _pro_dsc_profile(
        gamma=300,
        beta=400,
        pieta=0.1,
        piiter=5,
        z_dim=128,
        epochs=200,
        batch_size=1024,
        warmup_steps=-1,
    ),
    "places365standard": _pro_dsc_profile(
        gamma=800,
        beta=400,
        pieta=0.09,
        piiter=1,
        z_dim=1024,
        epochs=100,
        batch_size=2048,
        warmup_steps=2000,
    ),
    "dtd": _pro_dsc_profile(
        gamma=300,
        beta=400,
        pieta=0.1,
        piiter=1,
        z_dim=128,
        epochs=200,
        batch_size=512,
        warmup_steps=100,
    ),
    "aircraft": _pro_dsc_profile(
        gamma=200,
        beta=400,
        pieta=0.1,
        piiter=1,
        z_dim=256,
        epochs=200,
        batch_size=512,
        warmup_steps=100,
    ),
    "cars": _pro_dsc_profile(
        gamma=300,
        beta=400,
        pieta=0.1,
        piiter=1,
        z_dim=256,
        epochs=200,
        batch_size=1024,
        warmup_steps=200,
    ),
    "flowers": _pro_dsc_profile(
        gamma=200,
        beta=300,
        pieta=0.12,
        piiter=1,
        z_dim=256,
        epochs=200,
        batch_size=512,
        cluster_learning_rate=3.0e-4,
        warmup_steps=100,
    ),
    "food": _pro_dsc_profile(
        gamma=300,
        beta=400,
        pieta=0.1,
        piiter=1,
        z_dim=256,
        epochs=100,
        batch_size=1500,
        warmup_steps=200,
    ),
    "pets": _pro_dsc_profile(
        gamma=300,
        beta=400,
        pieta=0.1,
        piiter=1,
        z_dim=128,
        epochs=100,
        batch_size=512,
        warmup_steps=100,
    ),
    "ucf101": _pro_dsc_profile(
        gamma=200,
        beta=300,
        pieta=0.1,
        piiter=1,
        z_dim=256,
        epochs=100,
        batch_size=1024,
        warmup_steps=200,
    ),
}


PRO_DSC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}


PRO_DSC_BACKBONE_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb16", "cifar10"): {"gamma": 200, "beta": 500},
    ("laion400m", "vitb16", "cifar20"): {"gamma": 200, "beta": 500},
    ("laion400m", "vitb16", "cifar100"): {"gamma": 100, "beta": 300},
    ("laion400m", "vitb16", "dtd"): {"gamma": 100, "pieta": 0.15},
    ("laion400m", "vitb16", "flowers"): {"batch_size": 128},
    ("laion400m", "vitb16", "pets"): {"gamma": 200, "beta": 300},
    ("laion400m", "vitl14", "cifar20"): {"gamma": 200, "beta": 600},
    ("laion400m", "vitl14", "ucf101"): {"batch_size": 256},
    ("laion400m", "vitl14", "flowers"): {"batch_size": 256},
    ("laion400m", "vitl14", "food"): {"gamma": 200},
}


def _normalize_pro_dsc_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_pro_dsc_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in PRO_DSC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _uses_pro_dsc_generic_hyperparameter_fallback(openclip_pretraining: str | None) -> bool:
    return (
        openclip_pretraining is not None
        and _normalize_pro_dsc_profile_token(openclip_pretraining) == "siglip"
    )


def _resolve_pro_dsc_dataset_hyperparameter_defaults(dataset_name: str) -> dict[str, Any]:
    profile_key = _normalize_pro_dsc_profile_dataset(dataset_name)
    if profile_key in PRO_DSC_OFFICIAL_DEFAULTS:
        return dict(PRO_DSC_OFFICIAL_DEFAULTS[profile_key])
    return dict(PRO_DSC_BENCHMARK_DEFAULTS.get(profile_key, {}))


def _resolve_pro_dsc_backbone_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    if _uses_pro_dsc_generic_hyperparameter_fallback(openclip_pretraining):
        return {}
    profile_key = (
        _normalize_pro_dsc_profile_token(openclip_pretraining),
        _normalize_pro_dsc_profile_token(openclip_backbone),
        _normalize_pro_dsc_profile_dataset(dataset_name),
    )
    return dict(PRO_DSC_BACKBONE_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_pro_dsc_backbone_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    if _uses_pro_dsc_generic_hyperparameter_fallback(openclip_pretraining):
        return None
    profile_key = (
        _normalize_pro_dsc_profile_token(openclip_pretraining),
        _normalize_pro_dsc_profile_token(openclip_backbone),
        _normalize_pro_dsc_profile_dataset(dataset_name),
    )
    if profile_key not in PRO_DSC_BACKBONE_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_pro_dsc_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
    openclip_backbone: str | None = None,
) -> dict[str, Any]:
    if _uses_pro_dsc_generic_hyperparameter_fallback(openclip_pretraining):
        return {}
    defaults = _resolve_pro_dsc_dataset_hyperparameter_defaults(dataset_name)
    if openclip_pretraining is not None and openclip_backbone is not None:
        defaults.update(
            _resolve_pro_dsc_backbone_hyperparameter_defaults(
                openclip_pretraining,
                openclip_backbone,
                dataset_name,
            )
        )
    return defaults


def _resolve_pro_dsc_profile_source(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
) -> str | None:
    if _uses_pro_dsc_generic_hyperparameter_fallback(openclip_pretraining):
        return None
    profile_key = _normalize_pro_dsc_profile_dataset(dataset_name)
    if profile_key in PRO_DSC_OFFICIAL_DEFAULTS:
        return "official"
    if profile_key in PRO_DSC_BENCHMARK_DEFAULTS:
        return "benchmark"
    return None


@dataclass(slots=True)
class PRODSCOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class PRODSCNetwork(require_module("torch.nn", "pip install torch").Module):
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
        logits = self.cluster(pre_feature).float()
        z = torch.nn.functional.normalize(z, dim=1)
        logits = torch.nn.functional.normalize(logits, dim=1)
        return z, logits


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
        thresh = 1.0e-12
        error = None

        for index in range(self.max_iter):
            if index % 2 == 0:
                prev_u = u
                u = self.eps * (torch.log(mu) - torch.logsumexp(self._m(matrix, u, v), dim=-1)) + u
                error = (u - prev_u).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self._m(matrix, u, v).transpose(-2, -1), dim=-1)) + v
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1.0e8] = 0.0
                v = v.detach().requires_grad_(True)

            if error is not None and float(error.item()) < thresh:
                break

        pi = torch.exp(self._m(matrix, u, v))
        return pi, matrix, u, v

    def _m(self, cost, u, v):
        return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


def run_pro_dsc_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> PRODSCOutputs:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("pro_dsc", params)

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
        "pro_dsc",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="PRO-DSC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if eval_dataset.labels is None:
        raise ValueError("PRO-DSC requires labels on the evaluation split.")

    train_cluster_num = _resolve_train_cluster_num(params, train_dataset)
    eval_cluster_num = int(params.get("eval_n_clusters", eval_dataset.num_classes or train_cluster_num))
    if train_cluster_num <= 0:
        raise ValueError("PRO-DSC requires train_n_clusters > 0.")
    if eval_cluster_num <= 0:
        raise ValueError("PRO-DSC requires eval_n_clusters > 0.")

    profile_key = _normalize_pro_dsc_profile_dataset(train_dataset.name)
    dataset_profile = _resolve_pro_dsc_hyperparameter_defaults(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    ) or None
    profile_source = _resolve_pro_dsc_profile_source(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
    )
    backbone_profile_source = _resolve_pro_dsc_backbone_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    gamma = int(_resolve_param(params, "gamma", default=_profile_default(dataset_profile, "gamma", 300)))
    beta = int(_resolve_param(params, "beta", default=_profile_default(dataset_profile, "beta", 400)))
    pieta = float(_resolve_param(params, "pieta", default=_profile_default(dataset_profile, "pieta", 0.1)))
    piiter = int(_resolve_param(params, "piiter", default=_profile_default(dataset_profile, "piiter", 1)))
    image_batch_size = int(params.get("image_batch_size", 256))

    with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
        train_image_features, _ = load_or_compute_raw_image_embeddings(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )
    with progress.stage(f"Encoding evaluation image split '{eval_dataset.name}/{eval_split}'"):
        eval_image_features, eval_labels = load_or_compute_raw_image_embeddings(
            dataset=eval_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )

    input_dim = int(train_image_features.shape[1])
    hidden_dim = int(
        _resolve_param(
            params,
            "hidden_dim",
            default=_profile_default(dataset_profile, "hidden_dim", 4096),
        )
    )
    z_dim = int(
        _resolve_param(
            params,
            "z_dim",
            default=_profile_default(dataset_profile, "z_dim", _fallback_z_dim(train_cluster_num)),
        )
    )
    epochs = int(_resolve_param(params, "epochs", alias="epo", default=_profile_default(dataset_profile, "epochs", 100)))
    batch_size = int(
        _resolve_param(
            params,
            "batch_size",
            alias="bs",
            default=_profile_default(dataset_profile, "batch_size", 1500),
        )
    )
    learning_rate = float(
        _resolve_param(
            params,
            "learning_rate",
            alias="lr",
            default=_profile_default(dataset_profile, "learning_rate", 1.0e-4),
        )
    )
    cluster_learning_rate = float(
        _resolve_param(
            params,
            "cluster_learning_rate",
            alias="lr_c",
            default=_profile_default(dataset_profile, "cluster_learning_rate", 1.0e-4),
        )
    )
    momentum = float(
        _resolve_param(
            params,
            "momentum",
            alias="momo",
            default=_profile_default(dataset_profile, "momentum", 0.9),
        )
    )
    weight_decay_main = float(
        _resolve_param(
            params,
            "weight_decay_main",
            alias="wd1",
            default=_profile_default(dataset_profile, "weight_decay_main", 1.0e-4),
        )
    )
    weight_decay_cluster = float(
        _resolve_param(
            params,
            "weight_decay_cluster",
            alias="wd2",
            default=_profile_default(dataset_profile, "weight_decay_cluster", 5.0e-3),
        )
    )
    eps = float(_resolve_param(params, "eps", default=_profile_default(dataset_profile, "eps", 0.1)))
    warmup_steps = int(
        _resolve_param(
            params,
            "warmup_steps",
            alias="warmup",
            default=_profile_default(dataset_profile, "warmup_steps", -1),
        )
    )
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)
    save_every = checkpoint_interval(params, enabled=save_checkpoint, default=50)
    validate_every = int(params.get("validate_every", 25))
    spectral_kmeans_n_init = int(params.get("spectral_kmeans_n_init", 10))
    spectral_random_state = params.get("spectral_random_state", seed)
    spectral_solver_type = str(params.get("spectral_solver_type", "lm"))
    spectral_tol = float(params.get("spectral_tol", 0.0))
    normalize_spectral_embedding = bool(params.get("normalize_spectral_embedding", True))

    if batch_size <= 0:
        raise ValueError("PRO-DSC requires batch_size > 0.")
    if train_image_features.shape[0] < 2:
        raise ValueError("PRO-DSC training requires at least two training samples.")
    effective_batch_size = min(batch_size, train_image_features.shape[0])
    if effective_batch_size <= train_cluster_num:
        raise ValueError(
            f"PRO-DSC requires effective batch_size > train_n_clusters for the block-prior eigendecomposition, "
            f"got effective_batch_size={effective_batch_size} and train_n_clusters={train_cluster_num}."
        )
    if effective_batch_size != batch_size:
        progress.log(
            f"Reducing PRO-DSC batch size from {batch_size} to {effective_batch_size} "
            f"for {train_image_features.shape[0]} training sample(s)"
        )
    if eval_cluster_num >= eval_image_features.shape[0]:
        raise ValueError(
            f"PRO-DSC requires eval_n_clusters < number of evaluation samples, got {eval_cluster_num} and "
            f"{eval_image_features.shape[0]} samples."
        )

    train_tensor = torch.from_numpy(train_image_features.astype("float32"))
    train_loader = data_mod.DataLoader(
        data_mod.TensorDataset(train_tensor),
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"PRO-DSC training loader is empty. Increase train samples or reduce batch_size (current {batch_size})."
        )

    model = PRODSCNetwork(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim).to(device)
    sink_layer = SinkhornDistance(pieta, max_iter=piiter)
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
    hyperparameter_fallback_version = (
        PRO_DSC_GENERIC_FALLBACK_VERSION
        if _uses_pro_dsc_generic_hyperparameter_fallback(openclip_pretraining)
        else None
    )
    artifact_root = (
        cache_root / hyperparameter_fallback_version
        if hyperparameter_fallback_version is not None
        else cache_root
    )
    artifact_dir = ensure_dir(
        artifact_root
        / _artifact_token(
            source_dataset=train_dataset.name,
            train_split=train_split,
            evaluation_dataset=eval_dataset.name,
            evaluation_split=eval_split,
            train_cluster_num=train_cluster_num,
            eval_cluster_num=eval_cluster_num,
        )
    )
    checkpoint_dir = ensure_checkpoint_dir(artifact_dir) if save_checkpoint else None
    source_eval_cluster_num = train_dataset.num_classes or eval_cluster_num
    source_artifact_dir = artifact_root / _artifact_token(
        source_dataset=train_dataset.name,
        train_split=train_split,
        evaluation_dataset=train_dataset.name,
        evaluation_split="val",
        train_cluster_num=train_cluster_num,
        eval_cluster_num=source_eval_cluster_num,
    )
    load_checkpoint = load_checkpoint_enabled(params, train_dataset.name)
    existing_checkpoint = (
        first_existing_checkpoint(
            [
                path
                for path in (
                    latest_checkpoint(artifact_dir / "checkpoints", "pro_dsc_epoch_*.pt"),
                    latest_checkpoint(source_artifact_dir / "checkpoints", "pro_dsc_epoch_*.pt"),
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
        progress.log(f"Loading PRO-DSC checkpoint and skipping training: {existing_checkpoint}")
        payload = load_torch_checkpoint(existing_checkpoint, device=device)
        model.load_state_dict(payload["model"], strict=True)
        checkpoint_loaded = True
        loaded_checkpoint_path = existing_checkpoint

    warmup_step = 0
    cluster_head_bootstrapped = False
    training_history: list[dict[str, float | int | bool]] = []

    efficiency_measurement = None
    if not checkpoint_loaded:
        efficiency_measurement = start_train_eval_measurement(device)
        progress.log(f"Training PRO-DSC for {epochs} epoch(s)")
    for epoch_index in range(0 if checkpoint_loaded else epochs):
        model.train()
        epoch_tcr = 0.0
        epoch_exp = 0.0
        epoch_block = 0.0
        epoch_total = 0.0
        epoch_batches = 0
        warmup_active = warmup_step <= warmup_steps

        for (x_batch,) in train_loader:
            current_batch_size = int(x_batch.size(0))
            x_batch = x_batch.to(device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=use_amp):
                z, logits = model(x_batch)
                self_coeff = logits @ logits.T
                sign_self_coeff = torch.sign(self_coeff)
                projected_coeff = sink_layer(self_coeff.abs().unsqueeze(0))[0]
                projected_coeff = projected_coeff * projected_coeff.shape[-1]
                projected_coeff = projected_coeff[0]
                projected_coeff = projected_coeff - torch.diag(torch.diag(projected_coeff))

                affinity = 0.5 * (projected_coeff.abs() + projected_coeff.abs().T)
                laplacian = torch.diag(affinity.sum(1)) - affinity
                with torch.no_grad():
                    _, eigenvectors = torch.linalg.eigh(laplacian)
                    u_hat = eigenvectors[:, :train_cluster_num]
                    w_matrix = u_hat @ u_hat.T

                if warmup_step <= warmup_steps:
                    loss_tcr = warmup_criterion(z)
                    loss_exp = None
                    loss_block = None
                    loss = loss_tcr
                else:
                    loss_tcr = warmup_criterion(z)
                    loss_exp = (
                        0.5
                        * torch.linalg.norm(z.T - z.T @ sign_self_coeff.mul(affinity)) ** 2
                        / current_batch_size
                    )
                    loss_block = torch.trace(laplacian.T @ w_matrix) / current_batch_size
                    loss = loss_tcr + gamma * loss_exp + beta * loss_block

            if warmup_step <= warmup_steps:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                optimizer_cluster.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.step(optimizer_cluster)
                scaler.update()

            if warmup_step == warmup_steps and not cluster_head_bootstrapped:
                _update_cluster_from_subspace(model)
                cluster_head_bootstrapped = True

            epoch_total += float(loss.item())
            epoch_tcr += float(loss_tcr.item())
            if loss_exp is not None:
                epoch_exp += float(loss_exp.item())
            if loss_block is not None:
                epoch_block += float(loss_block.item())
            epoch_batches += 1
            warmup_step += 1

        epoch_record = {
            "epoch": epoch_index + 1,
            "warmup_active": warmup_active,
            "loss_total": epoch_total / max(epoch_batches, 1),
            "loss_tcr": epoch_tcr / max(epoch_batches, 1),
            "loss_exp": epoch_exp / max(epoch_batches, 1) if not warmup_active else 0.0,
            "loss_block": epoch_block / max(epoch_batches, 1) if not warmup_active else 0.0,
        }
        training_history.append(epoch_record)
        progress.epoch(
            "PRO-DSC training",
            epoch_index + 1,
            epochs,
            metrics={
                "loss": epoch_record["loss_total"],
                "warmup": warmup_active,
            },
        )

        if save_every > 0 and ((epoch_index + 1) % save_every == 0 or epoch_index + 1 == epochs):
            torch.save(
                {
                    "epoch": epoch_index + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_cluster": optimizer_cluster.state_dict(),
                },
                checkpoint_dir / f"pro_dsc_epoch_{epoch_index + 1}.pt",
            )

    with progress.stage(f"Running PRO-DSC spectral clustering with {eval_cluster_num} cluster(s)"):
        prediction_array = _predict_with_spectral_clustering(
            model=model,
            features=eval_image_features.astype("float32"),
            batch_size=batch_size,
            device=device,
            n_clusters=eval_cluster_num,
            kmeans_n_init=spectral_kmeans_n_init,
            random_state=None if spectral_random_state is None else int(spectral_random_state),
            solver_type=spectral_solver_type,
            tol=spectral_tol,
            normalize_embedding=normalize_spectral_embedding,
        )
    if efficiency_measurement is not None:
        efficiency_measurement.stop()

    return PRODSCOutputs(
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
            "train_n_clusters": train_cluster_num,
            "eval_n_clusters": eval_cluster_num,
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
            "weight_decay_main": weight_decay_main,
            "weight_decay_cluster": weight_decay_cluster,
            "gamma": gamma,
            "beta": beta,
            "pieta": pieta,
            "piiter": piiter,
            "eps": eps,
            "warmup_steps": warmup_steps,
            "save_checkpoint": save_checkpoint,
            "save_every": save_every,
            "load_checkpoint": load_checkpoint,
            "checkpoint_loaded": checkpoint_loaded,
            "loaded_checkpoint_path": None if loaded_checkpoint_path is None else str(loaded_checkpoint_path),
            "validate_every": validate_every,
            "spectral_kmeans_n_init": spectral_kmeans_n_init,
            "spectral_random_state": spectral_random_state,
            "spectral_solver_type": spectral_solver_type,
            "spectral_tol": spectral_tol,
            "normalize_spectral_embedding": normalize_spectral_embedding,
            "hyperparameter_profile": None if profile_source is None else profile_key,
            "hyperparameter_profile_source": profile_source,
            "backbone_hyperparameter_profile": backbone_profile_source,
            "hyperparameter_fallback_version": hyperparameter_fallback_version,
            "official_default_profile": profile_key if profile_source == "official" else None,
            "artifact_dir": str(artifact_dir),
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "train_feature_count": int(train_image_features.shape[0]),
            "evaluation_feature_count": int(eval_image_features.shape[0]),
            "shared_feature_cache": "common raw OpenCLIP image features",
            "training_history": training_history,
            "protocol_note": (
                "The official PRO-DSC repo trains on pre-extracted CLIP features and reports validation metrics during "
                "training. This benchmark version keeps the PRO-DSC training objective, switches the source features to "
                "shared OpenCLIP image embeddings, trains on the source-train split only, and produces final predictions "
                "from a single label-free spectral clustering run on the evaluation split."
            ),
        },
    )


def _predict_with_spectral_clustering(
    *,
    model: PRODSCNetwork,
    features: np.ndarray,
    batch_size: int,
    device,
    n_clusters: int,
    kmeans_n_init: int,
    random_state: int | None,
    solver_type: str,
    tol: float,
    normalize_embedding: bool,
) -> np.ndarray:
    torch = require_module("torch", "pip install torch")

    model.eval()
    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device=device, dtype=torch.float32)
            _, logits = model(batch)
            logits_list.append(logits.cpu().numpy().astype("float32"))
    logits_matrix = np.concatenate(logits_list, axis=0)
    self_coeff = np.abs(logits_matrix @ logits_matrix.T)
    return _spectral_cluster(self_coeff, n_clusters, kmeans_n_init, random_state, solver_type, tol, normalize_embedding)


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
        raise ValueError(f"Unknown PRO-DSC spectral solver_type '{solver_type}'.")

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


def _update_cluster_from_subspace(model: PRODSCNetwork) -> PRODSCNetwork:
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


def _resolve_train_cluster_num(params: dict[str, Any], train_dataset) -> int:
    explicit = params.get("train_n_clusters", params.get("n_clusters"))
    if explicit is not None:
        return int(explicit)
    if train_dataset.num_classes <= 0:
        raise ValueError("PRO-DSC needs train_n_clusters when the source dataset does not define class_names.")
    return int(train_dataset.num_classes)


def _resolve_param(params: dict[str, Any], key: str, *, alias: str | None = None, default: Any) -> Any:
    if key in params:
        return params[key]
    if alias is not None and alias in params:
        return params[alias]
    return default


def _profile_default(profile: dict[str, Any] | None, key: str, fallback: Any) -> Any:
    if profile is None:
        return fallback
    return profile.get(key, fallback)


def _fallback_z_dim(train_cluster_num: int) -> int:
    if train_cluster_num >= 1000:
        return 1024
    if train_cluster_num >= 100:
        return 256
    return 128


def _artifact_token(
    *,
    source_dataset: str,
    train_split: str,
    evaluation_dataset: str,
    evaluation_split: str,
    train_cluster_num: int,
    eval_cluster_num: int,
) -> str:
    source_token = _normalize_dataset_name(source_dataset)
    eval_token = _normalize_dataset_name(evaluation_dataset)
    return (
        f"{source_token}__{train_split}__to__{eval_token}__{evaluation_split}"
        f"__traink{train_cluster_num}__evalk{eval_cluster_num}"
    )
