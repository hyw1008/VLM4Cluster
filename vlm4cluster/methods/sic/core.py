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
    COMMON_WORDNET_PROMPT_BUILDERS,
    COMMON_WORDNET_PROMPT_TEMPLATE_KEY,
    _normalize_rows,
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


SIC_DATASET_HYPERPARAMETER_DEFAULTS: dict[str, dict[str, int | float | bool]] = {
    "cifar10": {"gamma_r": 500, "xi_a": 30, "topk": 20, "ce_weight": 0.1},
    "cifar20": {"gamma_r": 200, "xi_a": 20, "topk": 20, "ce_weight": 1.0},
    "cifar100": {"gamma_r": 500, "xi_a": 50, "topk": 20, "ce_weight": 1.0},
    "stl10": {"gamma_r": 200, "xi_a": 20, "topk": 20, "ce_weight": 1.0},
    "imagenet10": {"gamma_r": 500, "xi_a": 30, "topk": 20, "ce_weight": 1.0},
    "imagenetdogs": {"gamma_r": 1000, "xi_a": 50, "topk": 20, "ce_weight": 1.0},
    "dtd": {"gamma_r": 500, "xi_a": 30, "topk": 20, "ce_weight": 1.0},
    "ucf101": {"gamma_r": 500, "xi_a": 50, "topk": 20, "ce_weight": 1.0},
    "imagenet": {
        "gamma_r": 300,
        "gamma_u": 0.1,
        "xi_c": 0.5,
        "xi_a": 20,
        "topk": 20,
        "ce_weight": 0.1,
        "center_init": True,
        "train_batch_size": 4096,
        "entropy_weight": 1.0,
        "learning_rate": 3.0e-4,
    },
    "places365standard": {"gamma_r": 1000, "xi_a": 50, "topk": 50, "ce_weight": 1.0},
    "aircraft": {"gamma_r": 1000, "xi_a": 50, "topk": 20, "ce_weight": 0.1, "epochs": 200},
    "cars": {"gamma_r": 1000, "xi_a": 50, "topk": 20, "ce_weight": 0.05, "epochs": 200},
    "flowers": {"gamma_r": 500, "xi_c": 0.6, "xi_a": 10, "topk": 20, "ce_weight": 0.1, "epochs": 200},
    "food": {"gamma_r": 1000, "gamma_u": 0.05, "xi_c": 0.6, "xi_a": 20, "topk": 20, "ce_weight": 0.1, "epochs": 200},
    "pets": {"gamma_r": 1000, "gamma_u": 0.05, "xi_c": 0.6, "xi_a": 20, "topk": 20, "ce_weight": 0.1, "epochs": 200},
}


SIC_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}


SIC_BACKBONE_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, int | float | bool]] = {
    ("laion400m", "vitb32", "imagenet10"): {"xi_c": 0.8},
    ("laion400m", "vitb32", "imagenetdogs"): {"xi_c": 0.8},
    ("laion400m", "vitb32", "places365standard"): {"xi_c": 0.8},
    ("laion400m", "vitb16", "cifar100"): {"xi_c": 1.0},
    ("laion400m", "vitb16", "stl10"): {"learning_rate": 3.0e-4},
    ("laion400m", "vitb16", "imagenet10"): {"xi_c": 0.8, "learning_rate": 3.0e-4},
    ("laion400m", "vitb16", "imagenetdogs"): {"xi_c": 0.8, "learning_rate": 3.0e-4},
    ("laion400m", "vitb16", "dtd"): {"xi_c": 0.8},
    ("laion400m", "vitb16", "ucf101"): {"xi_c": 0.8, "learning_rate": 1.0e-3},
    ("laion400m", "vitb16", "places365standard"): {"xi_c": 0.8},
    ("laion400m", "vitb16", "flowers"): {"xi_c": 0.7},
    ("laion400m", "vitb16", "pets"): {"learning_rate": 3.0e-4},
    ("laion400m", "vitl14", "stl10"): {"learning_rate": 3.0e-4},
    ("laion400m", "vitl14", "imagenet10"): {"learning_rate": 3.0e-4},
    ("laion400m", "vitl14", "imagenetdogs"): {"learning_rate": 3.0e-4},
    ("laion400m", "vitl14", "dtd"): {"xi_c": 0.7},
    ("laion400m", "vitl14", "places365standard"): {"xi_c": 0.8},
    ("laion400m", "vitl14", "flowers"): {"xi_c": 0.7},
}


SIC_FALLBACK_HYPERPARAMETER_DEFAULTS: dict[str, int | float | bool] = {
    "gamma_r": 500,
    "gamma_u": 0.05,
    "xi_c": 0.9,
    "xi_a": 30,
    "topk": 20,
    "ce_weight": 1.0,
    "center_init": False,
    "train_batch_size": 128,
    "entropy_weight": 5.0,
    "learning_rate": 1.0e-4,
    "epochs": 100,
    "normalize_features": False,
}


SIC_GENERIC_OPENCLIP_DEFAULTS_CACHE_TOKEN = "generic-openclip-defaults-v1"


@dataclass(slots=True)
class SICOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


class SICClusterHead(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_dim: int, num_clusters: int, num_heads: int) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.cluster_head_i = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, num_clusters) for _ in range(num_heads)]
        )

    def forward(self, x, forward_pass: str = "head_i"):
        if forward_pass not in {"head_i", "output_i"}:
            raise ValueError(f"Invalid SIC forward pass '{forward_pass}'.")
        return [head(x) for head in self.cluster_head_i]


class SICNeighborDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, neighbor_indices: np.ndarray, seed: int | None = None) -> None:
        self.neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.neighbor_indices.shape[0])

    def __getitem__(self, index: int):
        neighbor_index = int(self.rng.choice(self.neighbor_indices[index]))
        return {"index": int(index), "n_index": neighbor_index}


def run_sic_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> SICOutputs:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("sic", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "sic",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="SIC",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if train_dataset.num_classes == 0:
        raise ValueError("SIC requires datasets with known class names on the source train split.")
    if eval_dataset.labels is None:
        raise ValueError("SIC requires labels on the evaluation split.")

    hyperparameter_defaults = _resolve_sic_hyperparameter_defaults(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    hyperparameter_default_profile = _resolve_sic_dataset_hyperparameter_profile_source(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
    )
    backbone_hyperparameter_profile = _resolve_sic_backbone_hyperparameter_profile_source(
        train_dataset.name,
        openclip_pretraining=openclip_pretraining,
        openclip_backbone=openclip_backbone,
    )
    cluster_num, cluster_num_source = resolve_source_default_cluster_num(
        params,
        train_dataset,
        method_name="SIC",
    )
    num_heads = int(params.get("num_heads", 1))
    image_batch_size = int(params.get("image_batch_size", 128))
    train_batch_size = int(
        params.get("train_batch_size", params.get("batch_size", hyperparameter_defaults["train_batch_size"]))
    )
    text_batch_size = int(params.get("text_batch_size", 2048))
    epochs = int(params.get("epochs", hyperparameter_defaults["epochs"]))
    learning_rate = float(params.get("learning_rate", params.get("lr", hyperparameter_defaults["learning_rate"])))
    weight_decay_default = _resolve_sic_weight_decay_default(train_dataset.name)
    weight_decay = float(params.get("weight_decay", weight_decay_default))
    gamma_u = float(params.get("gamma_u", hyperparameter_defaults["gamma_u"]))
    gamma_r = int(params.get("gamma_r", hyperparameter_defaults["gamma_r"]))
    xi_c = float(params.get("xi_c", hyperparameter_defaults["xi_c"]))
    xi_a = int(params.get("xi_a", hyperparameter_defaults["xi_a"]))
    topk = int(params.get("topk", hyperparameter_defaults["topk"]))
    entropy_weight = float(params.get("entropy_weight", hyperparameter_defaults["entropy_weight"]))
    ce_weight = float(params.get("ce_weight", hyperparameter_defaults["ce_weight"]))
    center_init = bool(params.get("center_init", hyperparameter_defaults["center_init"]))
    normalize_features = bool(params.get("normalize_features", hyperparameter_defaults["normalize_features"]))
    kmeans_n_init = int(params.get("kmeans_n_init", params.get("kmeans_nredo", 20)))
    kmeans_max_iter = int(params.get("kmeans_max_iter", params.get("kmeans_niter", 300)))
    random_state = params.get("random_state", params.get("seed"))
    seed = int(params.get("seed", 42))
    force_recompute_nouns = bool(params.get("force_recompute_nouns", False))
    force_recompute_neighbors = bool(params.get("force_recompute_neighbors", False))
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)

    if cluster_num <= 0:
        raise ValueError("SIC requires n_clusters > 0.")
    if num_heads <= 0:
        raise ValueError("SIC requires num_heads > 0.")
    if epochs <= 0:
        raise ValueError("SIC requires epochs > 0.")
    if gamma_r <= 0:
        raise ValueError("SIC requires gamma_r > 0.")
    if topk <= 0:
        raise ValueError("SIC requires topk > 0.")
    if xi_a <= 0:
        raise ValueError("SIC requires xi_a > 0.")
    if not 0 < xi_c <= 1:
        raise ValueError("SIC requires xi_c in (0, 1].")

    with progress.stage(f"Encoding train image split '{train_dataset.name}/{train_split}'"):
        train_image_features, train_labels = _encode_image_dataset_raw(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    with progress.stage(f"Encoding evaluation image split '{eval_dataset.name}/{eval_split}'"):
        eval_image_features, eval_labels = _encode_image_dataset_raw(
            dataset=eval_dataset,
            bundle=bundle,
            batch_size=image_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
            cache_dir=output_dir,
        )
    if normalize_features:
        progress.log("Normalizing SIC image features with L2 norm")
        train_image_features = _normalize_rows(train_image_features.astype("float32"))
        eval_image_features = _normalize_rows(eval_image_features.astype("float32"))
    feature_space_cache_suffix = _sic_feature_space_cache_suffix(normalize_features)
    train_feature_cache_key = cache_key(
        image_dataset_cache_key(train_dataset),
        bundle.spec.cache_key,
        feature_space_cache_suffix,
    )
    selected_head = int(params.get("selected_head", 0))
    if selected_head < 0 or selected_head >= num_heads:
        raise ValueError(f"SIC selected_head must be in [0, {num_heads - 1}], got {selected_head}.")
    checkpoint_name = _sic_checkpoint_name(
        train_feature_cache_key,
        cluster_num,
        num_heads,
        openclip_pretraining=openclip_pretraining,
    )
    checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    checkpoint_loaded = False
    if load_checkpoint_enabled(params, train_dataset.name) and checkpoint_path.exists():
        progress.log(f"Loading SIC checkpoint and skipping head training: {checkpoint_path}")
        prediction_array, training_history = _infer_sic_checkpoint(
            eval_image_features=eval_image_features,
            cluster_num=cluster_num,
            num_heads=num_heads,
            selected_head=selected_head,
            batch_size=train_batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        checkpoint_loaded = True
        return SICOutputs(
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
                "hyperparameter_default_profile": hyperparameter_default_profile,
                "backbone_hyperparameter_profile": backbone_hyperparameter_profile,
                "n_clusters": cluster_num,
                "n_clusters_source": cluster_num_source,
                "num_heads": num_heads,
                "selected_head": selected_head,
                "image_batch_size": image_batch_size,
                "train_batch_size": train_batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gamma_u": gamma_u,
                "gamma_r": gamma_r,
                "xi_c": xi_c,
                "xi_a": xi_a,
                "topk": topk,
                "entropy_weight": entropy_weight,
                "ce_weight": ce_weight,
                "center_init": center_init,
                "normalize_features": normalize_features,
                "epochs": epochs,
                "save_checkpoint": save_checkpoint,
                "checkpoint_path": str(checkpoint_path),
                "load_checkpoint": load_checkpoint_enabled(params, train_dataset.name),
                "checkpoint_loaded": checkpoint_loaded,
                "training_history": training_history,
            },
        )

    efficiency_measurement = start_train_eval_measurement(device)
    with progress.stage(f"Running SIC initial K-Means with {cluster_num} cluster(s)"):
        initial_assignments, initial_centers = _run_sic_kmeans(
            train_image_features,
            cluster_num=cluster_num,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=None if random_state is None else int(random_state),
            cache_path=output_dir
            / (
                f"{train_feature_cache_key}"
                f"__kmeans_{cluster_num}"
                f"__{FAISS_KMEANS_CACHE_TOKEN}__ninit{kmeans_n_init}__maxiter{kmeans_max_iter}"
                f"__seed{random_state}.npz"
            ),
            progress=progress,
        )

    wordnet_csv = _resolve_sic_wordnet_csv(params)
    with progress.stage(f"Encoding SIC noun prompts from {wordnet_csv}"):
        nouns, noun_embeddings = _encode_sic_nouns(
            csv_path=wordnet_csv,
            bundle=bundle,
            batch_size=text_batch_size,
            device=device,
            cache_dir=output_dir,
            force_recompute=force_recompute_nouns,
        )
    with progress.stage("Selecting SIC semantic noun space"):
        selected_noun_indices = _select_sic_nouns(
            initial_centers=initial_centers,
            noun_embeddings=noun_embeddings,
            gamma_r=gamma_r,
            gamma_u=gamma_u,
            text_batch_size=text_batch_size,
            cache_path=output_dir
            / (
                f"{train_dataset.name}__{train_split}__{bundle.spec.cache_key}{feature_space_cache_suffix}__sic_nouns"
                f"__{wordnet_source_token(wordnet_csv)}__{COMMON_WORDNET_PROMPT_TEMPLATE_KEY}"
                f"__k{cluster_num}__ninit{kmeans_n_init}__maxiter{kmeans_max_iter}__seed{random_state}"
                f"__{FAISS_KMEANS_CACHE_TOKEN}__gammar{gamma_r}__gammau{gamma_u:.4f}.npy"
            ),
            device=device,
        )
    if selected_noun_indices.size == 0:
        raise ValueError("SIC semantic filtering selected zero nouns. Please adjust gamma_u/gamma_r.")
    if xi_a > selected_noun_indices.size:
        raise ValueError(
            f"SIC xi_a ({xi_a}) exceeds the number of selected nouns ({selected_noun_indices.size})."
        )
    selected_nouns = [nouns[index] for index in selected_noun_indices.tolist()]
    selected_noun_embeddings = noun_embeddings[selected_noun_indices]

    with progress.stage(f"Computing SIC train KNN neighbors with topk={topk}"):
        train_neighbors = _compute_l2_neighbors(
            train_image_features,
            topk=topk,
            cache_path=output_dir
            / (
                f"{train_dataset.name}__{train_split}__{bundle.spec.cache_key}{feature_space_cache_suffix}"
                f"__neighbors_top{topk}__euclidean.npy"
            ),
            force_recompute=force_recompute_neighbors,
            progress=progress,
            label="SIC train-neighbor search",
        )

    model = SICClusterHead(train_image_features.shape[1], cluster_num, num_heads).to(device)
    if center_init:
        _initialize_cluster_heads(model, initial_centers, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_image_tensor = torch.from_numpy(train_image_features.astype("float32")).to(device)
    eval_image_tensor = torch.from_numpy(eval_image_features.astype("float32")).to(device)
    selected_noun_tensor = torch.from_numpy(selected_noun_embeddings.astype("float32")).to(device)

    train_loader = data_mod.DataLoader(
        SICNeighborDataset(train_neighbors, seed=seed),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    training_history: list[dict[str, float | int]] = []
    progress.log(f"Training SIC cluster head for {epochs} epoch(s)")
    for epoch_index in range(epochs):
        text_centers = _compute_adjusted_text_centers(
            image_features=train_image_tensor,
            selected_noun_embeddings=selected_noun_tensor,
            model=model,
            xi_c=xi_c,
            xi_a=xi_a,
            text_batch_size=text_batch_size,
            device=device,
        )
        model.train()
        epoch_total = 0.0
        epoch_consistency = 0.0
        epoch_entropy = 0.0
        epoch_ce = 0.0
        epoch_batches = 0

        for batch in train_loader:
            indices = batch["index"].to(device)
            neighbor_indices = batch["n_index"].to(device)
            anchor_features = train_image_tensor[indices]
            neighbor_features = train_image_tensor[neighbor_indices]

            anchor_outputs = model(anchor_features, forward_pass="head_i")
            neighbor_outputs = model(neighbor_features, forward_pass="head_i")

            total_losses = []
            consistency_losses = []
            entropy_losses = []
            ce_losses = []
            for anchor_logits, neighbor_logits in zip(anchor_outputs, neighbor_outputs):
                total_loss, consistency_loss, entropy_term, ce_term = _sic_loss(
                    image_output=anchor_logits,
                    image_nb_output=neighbor_logits,
                    image_feature=anchor_features,
                    text_center=text_centers,
                    epoch=epoch_index + 1,
                    entropy_weight=entropy_weight,
                    ce_weight=ce_weight,
                )
                total_losses.append(total_loss)
                consistency_losses.append(consistency_loss)
                entropy_losses.append(entropy_term)
                ce_losses.append(ce_term)

            optimizer.zero_grad()
            total = torch.stack(total_losses, dim=0).sum()
            total.backward()
            optimizer.step()

            epoch_total += float(np.mean([loss.item() for loss in total_losses]))
            epoch_consistency += float(np.mean([loss.item() for loss in consistency_losses]))
            epoch_entropy += float(np.mean([loss.item() for loss in entropy_losses]))
            epoch_ce += float(np.mean([loss.item() for loss in ce_losses]))
            epoch_batches += 1

        epoch_record = {
            "epoch": epoch_index + 1,
            "total_loss": epoch_total / max(epoch_batches, 1),
            "consistency_loss": epoch_consistency / max(epoch_batches, 1),
            "entropy_term": epoch_entropy / max(epoch_batches, 1),
            "ce_term": epoch_ce / max(epoch_batches, 1),
        }
        training_history.append(epoch_record)
        progress.epoch(
            "SIC training",
            epoch_index + 1,
            epochs,
            metrics={
                "loss": epoch_record["total_loss"],
                "consistency": epoch_record["consistency_loss"],
            },
        )

    if save_checkpoint:
        checkpoint_path = checkpoint_file(output_dir, checkpoint_name, create=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cluster_num": cluster_num,
                "num_heads": num_heads,
                "epochs": epochs,
                "history": training_history,
            },
            checkpoint_path,
        )

    if selected_head < 0 or selected_head >= num_heads:
        raise ValueError(f"SIC selected_head must be in [0, {num_heads - 1}], got {selected_head}.")

    model.eval()
    predictions: list[np.ndarray] = []
    with progress.stage(f"Running SIC evaluation with head {selected_head}"):
        with torch.no_grad():
            for start in range(0, eval_image_tensor.size(0), train_batch_size):
                batch = eval_image_tensor[start : start + train_batch_size]
                logits = model(batch, forward_pass="head_i")[selected_head]
                predictions.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64))
    prediction_array = np.concatenate(predictions, axis=0)
    efficiency_measurement.stop()

    return SICOutputs(
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
            "wordnet_csv": str(wordnet_csv),
            "n_clusters": cluster_num,
            "n_clusters_source": cluster_num_source,
            "hyperparameter_default_profile": hyperparameter_default_profile,
            "backbone_hyperparameter_profile": backbone_hyperparameter_profile,
            "num_heads": num_heads,
            "selected_head": selected_head,
            "image_batch_size": image_batch_size,
            "train_batch_size": train_batch_size,
            "text_batch_size": text_batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gamma_u": gamma_u,
            "gamma_r": gamma_r,
            "xi_c": xi_c,
            "xi_a": xi_a,
            "topk": topk,
            "entropy_weight": entropy_weight,
            "ce_weight": ce_weight,
            "center_init": center_init,
            "normalize_features": normalize_features,
            "kmeans_n_init": kmeans_n_init,
            "kmeans_max_iter": kmeans_max_iter,
            "train_image_count": int(train_image_features.shape[0]),
            "evaluation_image_count": int(eval_image_features.shape[0]),
            "selected_noun_count": int(selected_noun_indices.size),
            "selected_noun_examples": selected_nouns[:10],
            "initial_cluster_count": int(initial_centers.shape[0]),
            "initial_assignment_count": int(initial_assignments.shape[0]),
            "save_checkpoint": save_checkpoint,
            "checkpoint_path": str(checkpoint_path) if (checkpoint_loaded or save_checkpoint) else None,
            "load_checkpoint": load_checkpoint_enabled(params, train_dataset.name),
            "checkpoint_loaded": checkpoint_loaded,
            "training_history": training_history,
            "protocol_note": (
                "SIC is trained on the source train split only and evaluated once on the evaluation split. "
                "The original repo's evaluation-label early stopping is not used in this benchmark."
            ),
        },
    )


def _infer_sic_checkpoint(
    *,
    eval_image_features: np.ndarray,
    cluster_num: int,
    num_heads: int,
    selected_head: int,
    batch_size: int,
    device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    torch = require_module("torch", "pip install torch")
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    model = SICClusterHead(eval_image_features.shape[1], cluster_num, num_heads).to(device)
    model.load_state_dict(payload["model"], strict=True)
    eval_image_tensor = torch.from_numpy(eval_image_features.astype("float32")).to(device)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, eval_image_tensor.size(0), batch_size):
            batch = eval_image_tensor[start : start + batch_size]
            logits = model(batch, forward_pass="head_i")[selected_head]
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64))
    history = payload.get("history", [])
    return np.concatenate(predictions, axis=0), history if isinstance(history, list) else []


def _normalize_sic_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_sic_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in SIC_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _uses_sic_generic_openclip_defaults(openclip_pretraining: str | None) -> bool:
    return (
        openclip_pretraining is not None
        and _normalize_sic_profile_token(openclip_pretraining) == "siglip"
    )


def _sic_checkpoint_name(
    train_feature_cache_key: str,
    cluster_num: int,
    num_heads: int,
    *,
    openclip_pretraining: str,
) -> str:
    parts: list[str | int] = [train_feature_cache_key, f"k{cluster_num}", f"heads{num_heads}"]
    if _uses_sic_generic_openclip_defaults(openclip_pretraining):
        parts.append(SIC_GENERIC_OPENCLIP_DEFAULTS_CACHE_TOKEN)
    return f"{cache_key(*parts)}__sic_final.pt"


def _resolve_sic_weight_decay_default(dataset_name: str) -> float:
    return 0.009 if _normalize_dataset_name(dataset_name) == "imagenetdogs" else 1.0e-4


def _resolve_sic_dataset_hyperparameter_profile_source(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
) -> str | None:
    if _uses_sic_generic_openclip_defaults(openclip_pretraining):
        return None
    profile_key = _normalize_sic_profile_dataset(dataset_name)
    if profile_key not in SIC_DATASET_HYPERPARAMETER_DEFAULTS:
        return None
    return profile_key


def _resolve_sic_backbone_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> dict[str, int | float | bool]:
    if _uses_sic_generic_openclip_defaults(openclip_pretraining):
        return {}
    profile_key = (
        _normalize_sic_profile_token(openclip_pretraining),
        _normalize_sic_profile_token(openclip_backbone),
        _normalize_sic_profile_dataset(dataset_name),
    )
    return dict(SIC_BACKBONE_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_sic_backbone_hyperparameter_profile_source(
    dataset_name: str,
    *,
    openclip_pretraining: str,
    openclip_backbone: str,
) -> str | None:
    if _uses_sic_generic_openclip_defaults(openclip_pretraining):
        return None
    profile_key = (
        _normalize_sic_profile_token(openclip_pretraining),
        _normalize_sic_profile_token(openclip_backbone),
        _normalize_sic_profile_dataset(dataset_name),
    )
    if profile_key not in SIC_BACKBONE_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_sic_hyperparameter_defaults(
    dataset_name: str,
    *,
    openclip_pretraining: str | None = None,
    openclip_backbone: str | None = None,
) -> dict[str, int | float | bool]:
    if _uses_sic_generic_openclip_defaults(openclip_pretraining):
        return dict(SIC_FALLBACK_HYPERPARAMETER_DEFAULTS)
    normalized = _normalize_sic_profile_dataset(dataset_name)
    defaults = dict(SIC_FALLBACK_HYPERPARAMETER_DEFAULTS)
    defaults.update(SIC_DATASET_HYPERPARAMETER_DEFAULTS.get(normalized, {}))
    if openclip_pretraining is not None and openclip_backbone is not None:
        defaults.update(
            _resolve_sic_backbone_hyperparameter_defaults(
                dataset_name,
                openclip_pretraining=openclip_pretraining,
                openclip_backbone=openclip_backbone,
            )
        )
    return defaults


def _sic_feature_space_cache_suffix(normalize_features: bool) -> str:
    return "__l2norm" if normalize_features else ""


def _encode_image_dataset_raw(
    dataset,
    bundle: OpenCLIPBundle,
    batch_size: int,
    num_workers: int,
    device,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    del cache_dir
    return load_or_compute_raw_image_embeddings(
        dataset=dataset,
        bundle=bundle,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def _resolve_sic_wordnet_csv(params: dict[str, Any]) -> Path:
    return _resolve_tac_wordnet_csv(params)


def _encode_sic_nouns(
    csv_path: Path,
    bundle: OpenCLIPBundle,
    batch_size: int,
    device,
    cache_dir: Path,
    force_recompute: bool,
) -> tuple[list[str], np.ndarray]:
    del cache_dir
    nouns = _load_tac_wordnet_nouns(csv_path, None)
    _, _, ensemble_embeddings = load_or_compute_text_prompt_bank(
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
    return nouns, ensemble_embeddings


def _run_sic_kmeans(
    features: np.ndarray,
    cluster_num: int,
    n_init: int,
    max_iter: int,
    random_state: int | None,
    cache_path: Path,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        return payload["assignments"], payload["centers"]

    assignments, centers = run_faiss_kmeans(
        features,
        n_clusters=cluster_num,
        n_iter=max_iter,
        n_redo=n_init,
        spherical=False,
        random_state=random_state,
        progress=progress,
        label="SIC initial KMeans",
    )
    np.savez_compressed(cache_path, assignments=assignments, centers=centers)
    return assignments, centers


def _select_sic_nouns(
    initial_centers: np.ndarray,
    noun_embeddings: np.ndarray,
    gamma_r: int,
    gamma_u: float,
    text_batch_size: int,
    cache_path: Path,
    device=None,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    torch = require_module("torch", "pip install torch")
    center_tensor = torch.from_numpy(initial_centers.astype("float32"))
    noun_tensor = torch.from_numpy(noun_embeddings.astype("float32"))

    if gamma_r > noun_tensor.size(0):
        raise ValueError(f"SIC gamma_r ({gamma_r}) exceeds noun pool size ({noun_tensor.size(0)}).")
    filtered_by_centers = _topk_nouns_by_center_similarity(
        center_tensor,
        noun_tensor,
        gamma_r=gamma_r,
        batch_size=text_batch_size,
        device=device,
    )
    center_selected = filtered_by_centers.reshape(-1).cpu().numpy()

    noun_means = []
    for start in range(0, noun_tensor.size(0), text_batch_size):
        batch = noun_tensor[start : start + text_batch_size]
        noun_means.append(batch.mean(dim=0))
    global_mean = torch.stack(noun_means, dim=0).mean(dim=0, keepdim=True)
    similarity_to_mean = torch.cosine_similarity(noun_tensor, global_mean, dim=1)
    unique_selected = torch.where(similarity_to_mean < (1.0 - gamma_u))[0].cpu().numpy()

    selected = np.intersect1d(center_selected, unique_selected)
    np.save(cache_path, selected.astype(np.int64))
    return selected.astype(np.int64)


def _topk_nouns_by_center_similarity(center_tensor, noun_tensor, gamma_r: int, batch_size: int, device=None):
    torch = require_module("torch", "pip install torch")

    if batch_size <= 0:
        raise ValueError("SIC text_batch_size must be > 0.")

    compute_device = center_tensor.device if device is None else device
    center_tensor = torch.nn.functional.normalize(center_tensor.to(compute_device), dim=1, eps=1e-8)
    best_values = None
    best_indices = None

    for start in range(0, noun_tensor.size(0), batch_size):
        batch = noun_tensor[start : start + batch_size].to(compute_device)
        batch = torch.nn.functional.normalize(batch, dim=1, eps=1e-8)
        similarity = center_tensor @ batch.T
        local_k = min(gamma_r, similarity.size(1))
        candidate_values, candidate_indices = torch.topk(similarity, local_k, dim=1)
        candidate_indices = candidate_indices + start

        if best_values is None:
            best_values = candidate_values
            best_indices = candidate_indices
            continue

        merged_values = torch.cat([best_values, candidate_values], dim=1)
        merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
        keep_k = min(gamma_r, merged_values.size(1))
        best_values, keep_positions = torch.topk(merged_values, keep_k, dim=1)
        best_indices = torch.gather(merged_indices, 1, keep_positions)

    if best_indices is None:
        raise ValueError("SIC noun pool is empty.")
    return best_indices


def _compute_l2_neighbors(
    features: np.ndarray,
    topk: int,
    cache_path: Path,
    force_recompute: bool,
    progress=None,
    label: str | None = None,
) -> np.ndarray:
    if cache_path.exists() and not force_recompute:
        return np.load(cache_path)
    if topk >= features.shape[0]:
        raise ValueError(
            f"SIC topk ({topk}) must be smaller than the number of train samples ({features.shape[0]})."
        )

    _, neighbors = search_topk(
        features.astype("float32"),
        topk=topk + 1,
        faiss_metric="l2",
        sklearn_metric="euclidean",
        progress=progress,
        label=label,
    )
    neighbors = neighbors.astype(np.int64)

    np.save(cache_path, neighbors)
    return neighbors


def _initialize_cluster_heads(model: SICClusterHead, image_centers: np.ndarray, device) -> None:
    torch = require_module("torch", "pip install torch")
    alpha = 5 * 10e-3
    center_tensor = torch.tensor(image_centers, dtype=torch.float32, device=device)
    for head in model.cluster_head_i:
        head.weight.data = 2 * alpha * center_tensor
        head.bias.data = -alpha * torch.norm(center_tensor, dim=1) ** 2


def _compute_adjusted_text_centers(
    image_features,
    selected_noun_embeddings,
    model: SICClusterHead,
    xi_c: float,
    xi_a: int,
    text_batch_size: int,
    device,
):
    torch = require_module("torch", "pip install torch")
    image_scores = []
    batch_size = 1024
    model.eval()
    with torch.no_grad():
        for start in range(0, image_features.size(0), batch_size):
            batch = image_features[start : start + batch_size]
            outputs = model(batch, forward_pass="head_i")[0]
            image_scores.append(outputs)
    image_scores_tensor = torch.cat(image_scores, dim=0)
    image_centers = _compute_image_centers(image_features, image_scores_tensor, xi_c)
    image_centers = torch.nn.functional.normalize(image_centers.to(device), dim=1)

    nearest = _topk_nouns_by_center_similarity(
        image_centers,
        selected_noun_embeddings,
        gamma_r=xi_a,
        batch_size=text_batch_size,
        device=device,
    ).reshape(-1)
    nearest_text_embeddings = selected_noun_embeddings[nearest]
    text_centers = torch.cat(
        [
            nearest_text_embeddings[start : start + xi_a].mean(dim=0, keepdim=True)
            for start in range(0, nearest_text_embeddings.size(0), xi_a)
        ],
        dim=0,
    )
    return torch.nn.functional.normalize(text_centers, dim=1)


def _compute_image_centers(image_features, image_scores, xi_c: float):
    torch = require_module("torch", "pip install torch")
    num_per_cluster = image_scores.shape[0] // image_scores.shape[1]
    topk = int(num_per_cluster * xi_c)
    if topk <= 0:
        raise ValueError(
            "SIC adjusted center computation requires at least one confident sample per cluster. "
            "Please increase max_train_samples or xi_c."
        )
    _, idx_max = torch.topk(image_scores, topk, dim=0, largest=True, sorted=False)

    centers = []
    for cluster_index in range(idx_max.shape[1]):
        centers.append(image_features[idx_max[:, cluster_index], :].mean(dim=0, keepdim=True))
    return torch.cat(centers, dim=0)


def _sic_loss(
    image_output,
    image_nb_output,
    image_feature,
    text_center,
    epoch: int,
    entropy_weight: float,
    ce_weight: float,
):
    torch = require_module("torch", "pip install torch")

    image_prob = torch.softmax(image_output, dim=-1)
    image_nb_prob = torch.softmax(image_nb_output, dim=-1)
    similarity = torch.bmm(image_prob.unsqueeze(1), image_nb_prob.unsqueeze(2)).reshape(-1)
    consistency_loss = -torch.log(similarity.clamp(min=1e-8)).mean()

    entropy_value = _entropy(image_prob.mean(dim=0), input_as_probabilities=True)

    if epoch > 0:
        text_prob = torch.mm(image_feature, text_center.t()).softmax(dim=-1)
        pseudo_label = torch.argmax(text_prob, dim=1)
        ce_loss = torch.nn.functional.cross_entropy(image_output, pseudo_label, reduction="mean")
    else:
        ce_loss = torch.zeros((), dtype=image_output.dtype, device=image_output.device)

    total_loss = consistency_loss - entropy_weight * entropy_value + ce_weight * ce_loss
    return total_loss, consistency_loss, -entropy_weight * entropy_value, ce_weight * ce_loss


def _entropy(x, input_as_probabilities: bool):
    torch = require_module("torch", "pip install torch")
    eps = 1e-8
    if input_as_probabilities:
        x_ = torch.clamp(x, min=eps)
        value = x_ * torch.log(x_)
    else:
        value = torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)

    if value.ndim == 2:
        return -value.sum(dim=1).mean()
    if value.ndim == 1:
        return -value.sum()
    raise ValueError(f"Unsupported entropy tensor rank {value.ndim}.")
