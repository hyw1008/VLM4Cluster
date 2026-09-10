from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vlm4cluster.datasets import load_image_dataset
from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_file, first_existing_checkpoint
from vlm4cluster.methods.checkpointing import load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.checkpointing import save_checkpoint_enabled
from vlm4cluster.methods.feature_cache import (
    cache_key,
    image_dataset_cache_key,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
)
from vlm4cluster.methods.tac.core import _build_split_dataset, _encode_image_dataset, _normalize_rows
from vlm4cluster.models import load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import run_faiss_kmeans, search_topk
from vlm4cluster.utils.progress import get_progress_logger


IDC_DEFAULT_SPLITS: dict[str, tuple[str, str]] = {
    "cifar10": ("train", "test"),
    "cifar20": ("train", "test"),
    "cifar100": ("train", "test"),
    "stl10": ("train", "test"),
    "imagenet10": ("train", "val"),
    "imagenet_dogs": ("train", "val"),
    "dtd": ("trainval", "test"),
    "ucf101": ("train", "val"),
    "imagenet": ("train", "val"),
    "places365_standard": ("train", "val"),
    "aircraft": ("train", "test"),
    "cars": ("train", "test"),
    "flowers": ("train", "test"),
    "food": ("train", "test"),
    "pets": ("train", "test"),
}

IDC_EVAL_ONLY_SPLITS: dict[str, str] = {
    "imagenet_a": "test",
    "imagenet_sketch": "test",
    "imagenet_r": "test",
    "imagenet_v2": "test",
    "imagenet_c": "test",
}


DEFAULT_IDC_VALUE_BUDGET = 500
DEFAULT_IDC_LEARNING_RATE = 1.0e-3

IDC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

IDC_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar100"): {"value_budget": 2000},
    ("laion400m", "vitb32", "imagenet10"): {"value_budget": 3000},
    ("laion400m", "vitb32", "ucf101"): {"value_budget": 1000},
    ("laion400m", "vitb32", "imagenet"): {"value_budget": 2000},
    ("laion400m", "vitb32", "aircraft"): {"value_budget": 2000},
    ("laion400m", "vitb32", "cars"): {"value_budget": 2000},
    ("laion400m", "vitb32", "flowers"): {"value_budget": 1020},
    ("laion400m", "vitb32", "pets"): {"value_budget": 1000},
    ("laion400m", "vitb16", "cifar10"): {"value_budget": 100},
    ("laion400m", "vitb16", "cifar100"): {"value_budget": 5000},
    ("laion400m", "vitb16", "imagenet10"): {"value_budget": 3000},
    ("laion400m", "vitb16", "dtd"): {"value_budget": 1000},
    ("laion400m", "vitb16", "ucf101"): {"value_budget": 2000},
    ("laion400m", "vitb16", "imagenet"): {"value_budget": 2000},
    ("laion400m", "vitb16", "aircraft"): {"value_budget": 2000},
    ("laion400m", "vitb16", "cars"): {"value_budget": 2000},
    ("laion400m", "vitb16", "flowers"): {"value_budget": 1020},
    ("laion400m", "vitb16", "food"): {"value_budget": 1000},
    ("laion400m", "vitb16", "pets"): {"value_budget": 2000},
    ("laion400m", "vitl14", "cifar10"): {"learning_rate": 5.0e-3},
    ("laion400m", "vitl14", "cifar100"): {"value_budget": 3000},
    ("laion400m", "vitl14", "imagenet10"): {"value_budget": 3000},
    ("laion400m", "vitl14", "dtd"): {"value_budget": 1000},
    ("laion400m", "vitl14", "ucf101"): {"value_budget": 2000},
    ("laion400m", "vitl14", "imagenet"): {"value_budget": 2000},
    ("laion400m", "vitl14", "aircraft"): {"value_budget": 2000},
    ("laion400m", "vitl14", "cars"): {"value_budget": 2000},
    ("laion400m", "vitl14", "flowers"): {"value_budget": 1020},
    ("laion400m", "vitl14", "food"): {"value_budget": 1000},
    ("laion400m", "vitl14", "pets"): {"value_budget": 2000},
}


@dataclass(slots=True)
class IDCOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class IDCClusterHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, input_dim: int, n_clusters: int) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, n_clusters),
        )

    def forward(self, x):
        torch = require_module("torch", "pip install torch")
        return torch.nn.functional.softmax(self.net(x), dim=-1)


def run_idc_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> IDCOutputs:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("idc", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "idc",
        bundle.spec.cache_key,
    )

    source_dataset_name = inputs.image_dataset.name
    train_split_default, test_split_default = _resolve_default_splits(source_dataset_name)
    train_split = str(params.get("train_split", train_split_default))
    train_dataset = _build_split_dataset(inputs, train_split, params.get("max_train_samples"))

    target_dataset_name = params.get("target_dataset_name", params.get("target_dataset"))
    if target_dataset_name is None:
        eval_dataset = _build_split_dataset(
            inputs,
            str(params.get("test_split", test_split_default)),
            params.get("max_test_samples"),
        )
        evaluation_split = eval_dataset.split
    else:
        target_name = str(target_dataset_name)
        _, target_test_split_default = _resolve_default_splits(target_name, allow_eval_only=True)
        evaluation_split = str(params.get("target_test_split", params.get("test_split", target_test_split_default)))
        eval_dataset = _build_named_split_dataset(
            inputs,
            dataset_name=target_name,
            split=evaluation_split,
            max_samples=params.get("max_test_samples"),
        )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{evaluation_split}"
    )

    if train_dataset.num_classes == 0:
        raise ValueError("IDC requires datasets with known class names for the source train split.")
    if eval_dataset.labels is None:
        raise ValueError("IDC requires labels on the evaluation split.")

    hyperparameter_defaults = _resolve_idc_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    hyperparameter_profile_source = _resolve_idc_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    cluster_num = int(params.get("n_clusters", train_dataset.num_classes))
    eval_cluster_num = int(params.get("eval_n_clusters", cluster_num))
    eval_cluster_num_source = "explicit" if params.get("eval_n_clusters") is not None else "cluster_num"
    image_batch_size = int(params.get("image_batch_size", 256))
    value_budget = _resolve_idc_value_budget(params, hyperparameter_defaults)
    candidate_cluster_topk = int(params.get("candidate_cluster_topk", params.get("T", 5)))
    representativeness_k = int(params.get("representativeness_k", params.get("K", 20)))
    confidence_threshold = float(params.get("confidence_threshold", params.get("tau", 0.99)))
    warmup_epochs = int(params.get("warmup_epochs", 20))
    idc_warmup_epochs = int(params.get("idc_warmup_epochs", 50))
    train_epochs = int(params.get("epochs", 100))
    learning_rate = _resolve_idc_learning_rate(params, hyperparameter_defaults)
    batch_size_inquiry = int(params.get("batch_size_inquiry", 100))
    value_chunk_size = int(params.get("value_chunk_size", 500))
    pseudo_gamma = float(params.get("pseudo_gamma", 0.2))
    pseudo_chunk_size = int(params.get("pseudo_chunk_size", 2000))
    kmeans_niter = int(params.get("kmeans_niter", 300))
    kmeans_nredo = int(params.get("kmeans_nredo", 10))
    random_state = params.get("random_state", params.get("seed"))
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)

    if cluster_num <= 0:
        raise ValueError("IDC requires n_clusters > 0.")
    if value_budget <= 0:
        raise ValueError("IDC requires value_budget > 0.")
    if candidate_cluster_topk <= 0:
        raise ValueError("IDC requires candidate_cluster_topk > 0.")
    if representativeness_k <= 0:
        raise ValueError("IDC requires representativeness_k > 0.")
    if confidence_threshold <= 0 or confidence_threshold >= 1:
        raise ValueError("IDC requires confidence_threshold in (0, 1).")
    if value_budget > train_dataset.num_samples:
        raise ValueError(
            f"IDC value_budget ({value_budget}) exceeds the number of train samples ({train_dataset.num_samples})."
        )

    with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
        train_image_features, train_labels = _encode_image_dataset(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    with progress.stage(f"Encoding evaluation image split '{eval_dataset.name}/{evaluation_split}'"):
        eval_image_features, eval_labels = _encode_image_dataset(
            dataset=eval_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    if train_labels is None:
        raise ValueError("IDC requires labels on the train split to simulate oracle feedback.")

    checkpoint_name = f"{cache_key(image_dataset_cache_key(train_dataset), f'k{cluster_num}')}__idc_final.pt"
    checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    legacy_checkpoint_path = checkpoint_file(
        output_dir,
        f"{cache_key(train_dataset.name, train_split, train_dataset.name, 'val', f'k{cluster_num}')}__idc_final.pt",
    )
    legacy_target_checkpoint_path = checkpoint_file(
        output_dir,
        f"{cache_key(image_dataset_cache_key(train_dataset), image_dataset_cache_key(eval_dataset), f'k{cluster_num}')}__idc_final.pt",
    )
    load_checkpoint = load_checkpoint_enabled(params, train_dataset.name)
    existing_checkpoint = (
        first_existing_checkpoint([checkpoint_path, legacy_checkpoint_path, legacy_target_checkpoint_path])
        if load_checkpoint
        else None
    )
    checkpoint_loaded = False
    if load_checkpoint and existing_checkpoint is not None:
        progress.log(f"Loading IDC checkpoint and skipping head training: {existing_checkpoint}")
        predictions, training_history = _infer_idc_checkpoint(
            eval_image_features=eval_image_features,
            cluster_num=cluster_num,
            device=device,
            checkpoint_path=existing_checkpoint,
        )
        checkpoint_loaded = True
        return IDCOutputs(
            predictions=predictions.tolist(),
            evaluation_labels=eval_labels.tolist() if eval_labels is not None else None,
            evaluation_split=evaluation_split,
            metadata={
                "variant": "idc_openclip",
                "source_dataset": train_dataset.name,
                "evaluation_dataset": eval_dataset.name,
                "train_split": train_split,
                "evaluation_split": evaluation_split,
                "openclip_pretraining": bundle.spec.benchmark_pretraining,
                "openclip_backbone": bundle.spec.benchmark_backbone,
                "hyperparameter_default_profile": hyperparameter_profile_source,
                "cluster_num": cluster_num,
                "eval_cluster_num": eval_cluster_num,
                "eval_cluster_num_source": eval_cluster_num_source,
                "save_checkpoint": save_checkpoint,
                "checkpoint_path": str(existing_checkpoint),
                "load_checkpoint": load_checkpoint,
                "checkpoint_loaded": checkpoint_loaded,
                "training_history": training_history,
            },
        )

    efficiency_measurement = start_train_eval_measurement(device)
    with progress.stage(f"Running IDC K-Means initialization with {cluster_num} cluster(s)"):
        km_pred, km_centers = _run_kmeans_with_centers(
            train_image_features,
            cluster_num=cluster_num,
            n_iter=kmeans_niter,
            n_redo=kmeans_nredo,
            random_state=None if random_state is None else int(random_state),
        )
    km_metrics = _evaluate_predictions(train_labels, km_pred, cluster_num)

    model = IDCClusterHead(train_image_features.shape[1], cluster_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    z_all_t = torch.tensor(train_image_features, dtype=torch.float32, device=device)
    labels_t = torch.tensor(train_labels, dtype=torch.long, device=device)
    centers_t = torch.tensor(km_centers, dtype=torch.float32, device=device)
    km_t = torch.tensor(km_pred, dtype=torch.long, device=device)
    warmup_loader = data_mod.DataLoader(
        data_mod.TensorDataset(z_all_t, km_t),
        batch_size=512,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    progress.log(f"Warming up IDC cluster head for {warmup_epochs} epoch(s)")
    for warmup_epoch in range(warmup_epochs):
        model.train()
        for z_batch, y_batch in warmup_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = _loss_positive(model(z_batch), y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        progress.epoch("IDC warmup", warmup_epoch + 1, warmup_epochs)

    model.eval()
    with torch.no_grad():
        warmup_predictions = model(z_all_t).argmax(dim=1).cpu().numpy()
    warmup_metrics = _evaluate_predictions(train_labels, warmup_predictions, cluster_num)

    with progress.stage(f"Selecting {value_budget} valuable IDC samples"):
        selected_idx = _select_valuable_samples(
            z_all_t,
            centers_t,
            budget=value_budget,
            representativeness_k=representativeness_k,
            chunk=value_chunk_size,
        )
    with progress.stage("Simulating IDC oracle feedback on selected train samples"):
        pos_feedback, neg_feedback, neg_cands_t = _simulate_oracle(
            selected_idx,
            z_all_t,
            labels_t,
            centers_t,
            warmup_predictions,
            top_t=candidate_cluster_topk,
        )

    pos_indices = [sample for sample, _ in pos_feedback]
    pos_clusters = [cluster for _, cluster in pos_feedback]
    neg_indices = [sample for sample, _ in neg_feedback]
    z_pos_t = z_all_t[torch.tensor(pos_indices, device=device)] if pos_indices else None
    pos_tgt_t = torch.tensor(pos_clusters, dtype=torch.long, device=device) if pos_indices else None
    z_neg_t = z_all_t[torch.tensor(neg_indices, device=device)] if neg_indices else None
    pseudo_labels = -torch.ones(z_all_t.size(0), dtype=torch.long, device=device)
    training_history: list[dict[str, float | int]] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def sample_positive_batch():
        if z_pos_t is None or pos_tgt_t is None:
            return None, None
        index = torch.randperm(len(z_pos_t), device=device)[:batch_size_inquiry]
        return z_pos_t[index], pos_tgt_t[index]

    def sample_negative_batch():
        if z_neg_t is None or neg_cands_t is None:
            return None, None
        index = torch.randperm(len(z_neg_t), device=device)[:batch_size_inquiry]
        return z_neg_t[index], neg_cands_t[index]

    reg_chunk = 8192 if use_amp else pseudo_chunk_size
    progress.log(f"Fine-tuning IDC cluster head for {train_epochs} epoch(s)")
    for epoch in range(train_epochs):
        use_lreg = epoch >= idc_warmup_epochs
        if use_lreg:
            model.eval()
            _update_pseudo_labels(
                model=model,
                features=z_all_t,
                pseudo_labels=pseudo_labels,
                n_clusters=cluster_num,
                threshold_in=confidence_threshold,
                threshold_out=confidence_threshold,
                gamma=pseudo_gamma,
                chunk=reg_chunk,
            )
            model.train()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            lpos = z_all_t.new_tensor(0.0)
            pos_batch, pos_targets = sample_positive_batch()
            if pos_batch is not None and pos_targets is not None:
                lpos = _loss_positive(model(pos_batch), pos_targets, n_clusters=cluster_num)

            lneg = z_all_t.new_tensor(0.0)
            neg_batch, neg_candidates = sample_negative_batch()
            if neg_batch is not None and neg_candidates is not None:
                lneg = _loss_negative(model(neg_batch), neg_candidates)

            lreg = (
                _loss_regularisation(model, z_all_t, pseudo_labels, cluster_num)
                if use_lreg
                else z_all_t.new_tensor(0.0)
            )
            loss = lpos + lneg + lreg

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (epoch + 1) % 10 == 0 or epoch + 1 == train_epochs:
            model.eval()
            with torch.no_grad():
                epoch_predictions = model(z_all_t).argmax(dim=1).cpu().numpy()
            epoch_metrics = _evaluate_predictions(train_labels, epoch_predictions, cluster_num)
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "train_acc": epoch_metrics["acc"],
                    "train_ari": epoch_metrics["ari"],
                    "train_nmi": epoch_metrics["nmi"],
                    "pseudo_count": int((pseudo_labels != -1).sum().item()),
                }
            )
            model.train()
        progress.epoch(
            "IDC fine-tuning",
            epoch + 1,
            train_epochs,
            metrics={"loss": float(loss.item()), "lreg": use_lreg},
        )

    if save_checkpoint:
        checkpoint_path = checkpoint_file(output_dir, checkpoint_name, create=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cluster_num": cluster_num,
                "eval_cluster_num": eval_cluster_num,
                "warmup_epochs": warmup_epochs,
                "idc_warmup_epochs": idc_warmup_epochs,
                "epochs": train_epochs,
                "history": training_history,
            },
            checkpoint_path,
        )

    with progress.stage(f"Predicting evaluation split '{eval_dataset.name}/{evaluation_split}'"):
        z_eval = torch.tensor(eval_image_features, dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            predictions = model(z_eval).argmax(dim=1).cpu().numpy()
    efficiency_measurement.stop()

    return IDCOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=eval_labels.tolist() if eval_labels is not None else None,
        evaluation_split=evaluation_split,
        metadata={
            "variant": "idc_openclip",
            "source_dataset": train_dataset.name,
            "evaluation_dataset": eval_dataset.name,
            "train_split": train_split,
            "evaluation_split": evaluation_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "cluster_num": cluster_num,
            "eval_cluster_num": eval_cluster_num,
            "eval_cluster_num_source": eval_cluster_num_source,
            "train_image_count": int(train_image_features.shape[0]),
            "evaluation_image_count": int(eval_image_features.shape[0]),
            "value_budget": value_budget,
            "candidate_cluster_topk": candidate_cluster_topk,
            "representativeness_k": representativeness_k,
            "confidence_threshold": confidence_threshold,
            "warmup_epochs": warmup_epochs,
            "idc_warmup_epochs": idc_warmup_epochs,
            "epochs": train_epochs,
            "learning_rate": learning_rate,
            "batch_size_inquiry": batch_size_inquiry,
            "pseudo_gamma": pseudo_gamma,
            "kmeans_niter": kmeans_niter,
            "kmeans_nredo": kmeans_nredo,
            "kmeans_train_metrics": km_metrics,
            "warmup_train_metrics": warmup_metrics,
            "selected_sample_count": len(selected_idx),
            "positive_feedback_count": len(pos_feedback),
            "negative_feedback_count": len(neg_feedback),
            "save_checkpoint": save_checkpoint,
            "checkpoint_path": str(checkpoint_path) if (checkpoint_loaded or save_checkpoint) else None,
            "load_checkpoint": load_checkpoint,
            "checkpoint_loaded": checkpoint_loaded,
            "training_history": training_history,
        },
    )


def _normalize_idc_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_idc_profile_dataset(name: str) -> str:
    normalized = _normalize_idc_profile_token(name)
    if normalized in IDC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _resolve_idc_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    profile_key = (
        _normalize_idc_profile_token(openclip_pretraining),
        _normalize_idc_profile_token(openclip_backbone),
        _normalize_idc_profile_dataset(dataset_name),
    )
    return dict(IDC_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_idc_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    profile_key = (
        _normalize_idc_profile_token(openclip_pretraining),
        _normalize_idc_profile_token(openclip_backbone),
        _normalize_idc_profile_dataset(dataset_name),
    )
    if profile_key not in IDC_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_idc_value_budget(params: dict[str, Any], defaults: dict[str, Any]) -> int:
    if "value_budget" in params:
        return int(params["value_budget"])
    if "M" in params:
        return int(params["M"])
    return int(defaults.get("value_budget", DEFAULT_IDC_VALUE_BUDGET))


def _resolve_idc_learning_rate(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "learning_rate" in params:
        return float(params["learning_rate"])
    if "lr" in params:
        return float(params["lr"])
    return float(defaults.get("learning_rate", DEFAULT_IDC_LEARNING_RATE))


def _resolve_default_splits(dataset_name: str, allow_eval_only: bool = False) -> tuple[str, str]:
    normalized = dataset_name.strip().lower().replace("-", "_")
    if normalized in IDC_DEFAULT_SPLITS:
        return IDC_DEFAULT_SPLITS[normalized]
    if allow_eval_only and normalized in IDC_EVAL_ONLY_SPLITS:
        split = IDC_EVAL_ONLY_SPLITS[normalized]
        return "train", split
    raise ValueError(f"IDC does not define default split rules for dataset '{dataset_name}'.")


def _infer_idc_checkpoint(
    *,
    eval_image_features: np.ndarray,
    cluster_num: int,
    device,
    checkpoint_path,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    torch = require_module("torch", "pip install torch")
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    model = IDCClusterHead(eval_image_features.shape[1], cluster_num).to(device)
    model.load_state_dict(payload["model"], strict=True)
    z_eval = torch.tensor(eval_image_features, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        predictions = model(z_eval).argmax(dim=1).cpu().numpy()
    history = payload.get("history", [])
    return predictions.astype(np.int64, copy=False), history if isinstance(history, list) else []


def _build_named_split_dataset(
    inputs: MethodInputs,
    dataset_name: str,
    split: str,
    max_samples: Any,
):
    dataset_download = (
        inputs.image_dataset_config.download
        if dataset_name.strip().lower().replace("-", "_") == inputs.image_dataset_config.name.strip().lower().replace("-", "_")
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


def _run_kmeans_with_centers(
    matrix: np.ndarray,
    cluster_num: int,
    n_iter: int,
    n_redo: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalize_rows(matrix.astype("float32"))
    assignments, centers = run_faiss_kmeans(
        normalized,
        n_clusters=cluster_num,
        n_iter=n_iter,
        n_redo=n_redo,
        spherical=True,
        random_state=random_state,
    )
    centers = _normalize_rows(centers.astype("float32"))
    return assignments, centers


def _select_valuable_samples(
    features,
    centers,
    budget: int,
    representativeness_k: int,
    chunk: int,
) -> np.ndarray:
    hardness = _compute_hardness(features, centers)
    representativeness = _compute_representativeness(features, representativeness_k, chunk)
    hardness = (hardness - hardness.mean()) / (hardness.std() + 1e-8)
    representativeness = (representativeness - representativeness.mean()) / (representativeness.std() + 1e-8)

    diversity = features.new_zeros(features.size(0))
    value = hardness + representativeness + diversity
    active = require_module("torch", "pip install torch").ones(features.size(0), dtype=bool, device=features.device)
    selected: list[int] = []

    for step in range(budget):
        del step
        masked_value = value.masked_fill(~active, float("-inf"))
        best = int(masked_value.argmax().item())
        selected.append(best)
        active[best] = False
        similarity = features @ features[best]
        new_diversity = require_module("torch", "pip install torch").log((1.0 - similarity).clamp(min=1e-8))
        diversity = require_module("torch", "pip install torch").minimum(diversity, new_diversity) if selected[:-1] else new_diversity
        value = hardness + representativeness + diversity
    return np.asarray(selected, dtype=np.int64)


def _compute_hardness(features, centers):
    torch = require_module("torch", "pip install torch")
    similarity = features @ centers.t()
    top2, _ = torch.topk(similarity, 2, dim=1)
    return torch.log((1.0 - top2[:, 0] + top2[:, 1]).clamp(min=1e-8))


def _compute_representativeness(features, k: int, chunk: int):
    torch = require_module("torch", "pip install torch")
    n_samples, _ = features.shape
    effective_k = min(k, max(n_samples - 1, 1))

    if features.is_cuda:
        safe_chunk = max(16, int(512 * 1024 ** 2 // (n_samples * 4)))
        chunk = min(chunk, safe_chunk)
        knn_sum = torch.zeros(n_samples, device=features.device, dtype=features.dtype)
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            similarity = features[start:end] @ features.t()
            distances = (2.0 - 2.0 * similarity).clamp(min=0)
            local_indices = torch.arange(end - start, device=features.device)
            distances[local_indices, local_indices + start] = float("inf")
            nearest_distances, _ = torch.topk(distances, effective_k, dim=1, largest=False)
            knn_sum[start:end] = nearest_distances.sum(dim=1)
        return -torch.log(knn_sum.clamp(min=1e-8))

    normalized = features.detach().cpu().numpy().astype("float32")
    try:
        similarities, _ = search_topk(
            normalized,
            topk=effective_k + 1,
            faiss_metric="ip",
            sklearn_metric="cosine",
        )
        knn_similarity = similarities[:, 1:]
        knn_distance = np.clip(2.0 - 2.0 * knn_similarity, 0, None)
        knn_sum = knn_distance.sum(axis=1).astype(np.float32)
        result = -np.log(np.clip(knn_sum, 1e-8, None))
        return torch.tensor(result, dtype=features.dtype, device=features.device)
    except RuntimeError:
        pass

    knn_sum = torch.zeros(n_samples, device=features.device, dtype=features.dtype)
    for start in range(0, n_samples, chunk):
        end = min(start + chunk, n_samples)
        similarity = features[start:end] @ features.t()
        distances = (2.0 - 2.0 * similarity).clamp(min=0)
        local_indices = torch.arange(end - start, device=features.device)
        distances[local_indices, local_indices + start] = float("inf")
        nearest_distances, _ = torch.topk(distances, effective_k, dim=1, largest=False)
        knn_sum[start:end] = nearest_distances.sum(dim=1)
    return -torch.log(knn_sum.clamp(min=1e-8))


def _simulate_oracle(
    selected_idx: np.ndarray,
    features,
    labels,
    centers,
    predictions: np.ndarray,
    top_t: int,
):
    torch = require_module("torch", "pip install torch")
    from scipy.optimize import linear_sum_assignment

    labels_np = labels.cpu().numpy()
    n_clusters = centers.shape[0]
    n_labels = int(labels.max().item()) + 1
    size = max(n_clusters, n_labels)
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (predictions.astype(int), labels_np.astype(int)), 1)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    gt_to_cluster = np.zeros(n_labels, dtype=np.int64)
    for cluster_id, gt_id in zip(row_ind, col_ind):
        if gt_id < n_labels:
            gt_to_cluster[gt_id] = cluster_id
    label_adjusted = gt_to_cluster[labels_np]

    selected_tensor = torch.tensor(selected_idx, dtype=torch.long, device=features.device)
    similarity = features[selected_tensor] @ centers.t()
    top_candidates = torch.argsort(-similarity, dim=1)[:, :top_t]

    positive_feedback: list[tuple[int, int]] = []
    negative_feedback: list[tuple[int, list[int]]] = []
    for row, sample_index in enumerate(selected_idx.tolist()):
        true_cluster = int(label_adjusted[sample_index])
        candidates = top_candidates[row].tolist()
        if true_cluster in candidates:
            positive_feedback.append((sample_index, true_cluster))
        else:
            negative_feedback.append((sample_index, candidates))

    negative_candidates = None
    if negative_feedback:
        negative_candidates = torch.tensor(
            [candidates for _, candidates in negative_feedback],
            dtype=torch.long,
            device=features.device,
        )
    return positive_feedback, negative_feedback, negative_candidates


def _loss_positive(probabilities, targets, n_clusters: int | None = None):
    torch = require_module("torch", "pip install torch")
    if n_clusters is not None:
        index, counts = torch.unique(targets, return_counts=True)
        weight = torch.ones(n_clusters, device=targets.device, dtype=probabilities.dtype)
        weight[index] = targets.shape[0] / counts.float()
        return torch.nn.functional.nll_loss(torch.log(probabilities.clamp(min=1e-8)), targets, weight=weight)
    return torch.nn.functional.nll_loss(torch.log(probabilities.clamp(min=1e-8)), targets)


def _loss_negative(probabilities, negative_candidates):
    torch = require_module("torch", "pip install torch")
    n_samples, top_t = negative_candidates.shape
    random_column = torch.randint(0, top_t, (n_samples,), device=negative_candidates.device)
    chosen = negative_candidates[torch.arange(n_samples, device=negative_candidates.device), random_column]
    chosen_probability = probabilities[torch.arange(n_samples, device=probabilities.device), chosen]
    return -torch.log(1.0 - chosen_probability.clamp(max=1 - 1e-6)).mean()


def _update_pseudo_labels(
    model,
    features,
    pseudo_labels,
    n_clusters: int,
    threshold_in: float,
    threshold_out: float,
    gamma: float,
    chunk: int,
):
    torch = require_module("torch", "pip install torch")
    n_samples = features.size(0)
    per_class_limit = max(1, int(np.ceil(n_samples / n_clusters * gamma)))

    with torch.no_grad():
        probabilities = []
        for start in range(0, n_samples, chunk):
            probabilities.append(model(features[start : start + chunk]))
        probabilities = torch.cat(probabilities, dim=0)

    confidence, prediction = probabilities.max(dim=1)
    unconfident = confidence < threshold_out
    pseudo_new = -torch.ones(n_samples, dtype=torch.long, device=features.device)
    for cluster_index in range(n_clusters):
        member_index = (prediction == cluster_index).nonzero(as_tuple=False).squeeze(1)
        if member_index.numel() == 0:
            continue
        member_confidence = confidence[member_index]
        keep_num = min(member_index.numel(), per_class_limit)
        order = torch.argsort(-member_confidence)
        for rank in range(keep_num):
            sample_index = member_index[order[rank]].item()
            if confidence[sample_index] > threshold_in:
                pseudo_new[sample_index] = cluster_index
            else:
                break

    assignable = pseudo_labels == -1
    pseudo_labels[assignable] = pseudo_new[assignable]
    pseudo_labels[unconfident] = -1
    return pseudo_labels


def _loss_regularisation(model, features, pseudo_labels, n_clusters: int):
    torch = require_module("torch", "pip install torch")
    confident_mask = pseudo_labels != -1
    if int(confident_mask.sum()) == 0:
        return features.new_tensor(0.0)
    confident_features = features[confident_mask]
    confident_labels = pseudo_labels[confident_mask]
    probabilities = model(confident_features)
    index, counts = torch.unique(confident_labels, return_counts=True)
    weight = torch.ones(n_clusters, device=features.device, dtype=confident_features.dtype)
    weight[index] = confident_labels.shape[0] / counts.float()
    return torch.nn.functional.nll_loss(torch.log(probabilities.clamp(min=1e-8)), confident_labels, weight=weight)


def _evaluate_predictions(labels: np.ndarray, predictions: np.ndarray, n_clusters: int) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    size = max(n_clusters, int(predictions.max()) + 1, int(labels.max()) + 1)
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (labels.astype(int), predictions.astype(int)), 1)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    accuracy = contingency[row_ind, col_ind].sum() / len(labels)
    return {
        "acc": float(accuracy),
        "ari": float(adjusted_rand_score(labels, predictions)),
        "nmi": float(normalized_mutual_info_score(labels, predictions, average_method="arithmetic")),
    }
