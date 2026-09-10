from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label
from vlm4cluster.methods.base import MethodInputs
from vlm4cluster.methods.checkpointing import checkpoint_file, first_existing_checkpoint
from vlm4cluster.methods.checkpointing import load_checkpoint_enabled, load_torch_checkpoint
from vlm4cluster.methods.checkpointing import save_checkpoint_enabled
from vlm4cluster.methods.cluster_defaults import resolve_source_default_cluster_num
from vlm4cluster.methods.feature_cache import (
    cache_key,
    image_dataset_cache_key,
    load_or_compute_raw_image_embeddings,
    method_feature_cache_dir,
    resolve_benchmark_data_root,
)
from vlm4cluster.methods.scan.losses import ConfidenceBasedCE, SCANLoss, SimCLRLoss, entropy
from vlm4cluster.methods.tac.core import _resolve_domain_shift_datasets
from vlm4cluster.models import OpenCLIPBundle, load_openclip_bundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import start_train_eval_measurement
from vlm4cluster.utils.faiss_utils import search_topk
from vlm4cluster.utils.io import ensure_dir
from vlm4cluster.utils.progress import get_progress_logger


@dataclass(slots=True)
class SCANOutputs:
    predictions: list[int]
    evaluation_labels: list[int] | None
    evaluation_split: str
    metadata: dict[str, Any]


def _normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _uses_large_scale_scan_heads(name: str) -> bool:
    return _normalize_dataset_name(name) in {"imagenet", "places365standard"}


SCAN_HEADS_PER_DATASET = {
    "cifar10": 1,
    "cifar20": 1,
    "cifar100": 1,
    "stl10": 1,
    "imagenet10": 1,
    "imagenetdogs": 1,
}
DEFAULT_SCAN_NUM_HEADS = 10

SCAN_KNN_PER_DATASET = {
    "imagenet": 20,
    "imagenetv2": 20,
    "imagenetsketch": 20,
    "imageneta": 20,
    "imagenetr": 20,
    "imagenetc": 20,
    "places365standard": 20,
    "dtd": 50,
    "ucf101": 50,
    "cars": 50,
}
DEFAULT_SCAN_NUM_NEIGHBORS = 20
DEFAULT_SCAN_EPOCHS = 100

SCAN_IMAGENET_VARIANT_PROFILE_KEYS = {
    "imageneta",
    "imagenetsketch",
    "imagenetr",
    "imagenetv2",
    "imagenetc",
}

SCAN_OPT_PER_DATASET: dict[str, dict[str, float | int | str]] = {
    "cifar10": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "cifar20": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "cifar100": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "stl10": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "imagenet10": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "imagenetdogs": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "dtd": {"type": "adam", "lr": 3.0e-3, "wd": 1.0e-4, "momentum": 0.0, "batch": 32},
    "flowers": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "pets": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128},
    "ucf101": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 32},
    "aircraft": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 256},
    "cars": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 256},
    "food": {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 512},
    "imagenet": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
    "imagenetv2": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
    "imagenetsketch": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
    "imageneta": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
    "imagenetr": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
    "imagenetc": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
    "places365standard": {"type": "sgd", "lr": 30.0, "wd": 0.0, "momentum": 0.9, "batch": 4096},
}
DEFAULT_SCAN_OPT = {"type": "adam", "lr": 1.0e-4, "wd": 1.0e-4, "momentum": 0.0, "batch": 128}

SCAN_GENERIC_OPENCLIP_DEFAULTS: dict[str, Any] = {
    "num_neighbors": DEFAULT_SCAN_NUM_NEIGHBORS,
    "num_heads": DEFAULT_SCAN_NUM_HEADS,
    "entropy_weight": 5.0,
    "scan_epochs": DEFAULT_SCAN_EPOCHS,
    "scan_batch_size": DEFAULT_SCAN_OPT["batch"],
    "scan_optimizer": DEFAULT_SCAN_OPT["type"],
    "scan_learning_rate": DEFAULT_SCAN_OPT["lr"],
    "scan_weight_decay": DEFAULT_SCAN_OPT["wd"],
    "scan_momentum": DEFAULT_SCAN_OPT["momentum"],
}
SCAN_GENERIC_OPENCLIP_CACHE_TOKEN = "generic-openclip-defaults-v2"

SCAN_HYPERPARAMETER_DEFAULTS: dict[tuple[str, str, str], dict[str, Any]] = {
    ("laion400m", "vitb32", "cifar10"): {"scan_learning_rate": 3.0e-3},
    ("laion400m", "vitb32", "cifar100"): {"num_heads": 10, "entropy_weight": 10.0},
    ("laion400m", "vitb32", "ucf101"): {"num_neighbors": 10},
    ("laion400m", "vitb32", "cars"): {"scan_learning_rate": 1.0e-2},
    ("laion400m", "vitb32", "flowers"): {"scan_learning_rate": 3.0e-3},
    ("laion400m", "vitb32", "pets"): {"scan_learning_rate": 5.0e-3},
    ("laion400m", "vitb16", "cifar20"): {"scan_learning_rate": 3.0e-3},
    ("laion400m", "vitb16", "cifar100"): {"num_heads": 10, "entropy_weight": 10.0},
    ("laion400m", "vitb16", "stl10"): {"scan_learning_rate": 1.0e-3},
    ("laion400m", "vitb16", "imagenetdogs"): {"scan_learning_rate": 1.0e-3},
    ("laion400m", "vitb16", "ucf101"): {"num_neighbors": 10},
    ("laion400m", "vitb16", "cars"): {"scan_learning_rate": 1.0e-2},
    ("laion400m", "vitb16", "flowers"): {"scan_learning_rate": 3.0e-3},
    ("laion400m", "vitb16", "pets"): {"scan_learning_rate": 5.0e-3},
    ("laion400m", "vitl14", "cifar100"): {"num_heads": 10, "entropy_weight": 10.0},
    ("laion400m", "vitl14", "stl10"): {"scan_learning_rate": 1.0e-3},
    ("laion400m", "vitl14", "dtd"): {"scan_learning_rate": 1.0e-3},
    ("laion400m", "vitl14", "ucf101"): {"num_neighbors": 10},
    ("laion400m", "vitl14", "cars"): {"scan_learning_rate": 1.0e-2},
    ("laion400m", "vitl14", "flowers"): {"scan_learning_rate": 3.0e-3},
    ("laion400m", "vitl14", "pets"): {"scan_learning_rate": 5.0e-3},
}


def _default_scan_num_heads(name: str) -> int:
    return int(SCAN_HEADS_PER_DATASET.get(_normalize_dataset_name(name), DEFAULT_SCAN_NUM_HEADS))


def _default_scan_num_neighbors(name: str) -> int:
    return int(SCAN_KNN_PER_DATASET.get(_normalize_dataset_name(name), DEFAULT_SCAN_NUM_NEIGHBORS))


def _default_scan_epochs(name: str) -> int:
    return DEFAULT_SCAN_EPOCHS


def _default_scan_optimizer_config(name: str) -> dict[str, float | int | str]:
    return dict(SCAN_OPT_PER_DATASET.get(_normalize_dataset_name(name), DEFAULT_SCAN_OPT))


def _default_scan_profile_name(name: str) -> str:
    key = _normalize_dataset_name(name)
    if key in {"imagenet", "imagenetv2", "imagenetsketch", "imageneta", "imagenetr", "imagenetc"}:
        return "imagenet-scale"
    if key == "places365standard":
        return "places365-large-scale"
    if key in {"cifar10", "cifar20", "cifar100", "stl10", "imagenet10", "imagenetdogs"}:
        return "small"
    if key in {"dtd", "flowers", "pets", "ucf101", "aircraft", "cars", "food"}:
        return "fine-grained"
    return "fallback"


def _normalize_scan_profile_token(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")


def _normalize_scan_profile_dataset(name: str) -> str:
    normalized = _normalize_dataset_name(name)
    if normalized in SCAN_IMAGENET_VARIANT_PROFILE_KEYS:
        return "imagenet"
    return normalized


def _uses_scan_generic_openclip_defaults(
    openclip_pretraining: str,
    _openclip_backbone: str,
) -> bool:
    return _normalize_scan_profile_token(openclip_pretraining) == "siglip"


def _resolve_scan_generic_openclip_defaults(dataset_name: str | None = None) -> dict[str, Any]:
    defaults = dict(SCAN_GENERIC_OPENCLIP_DEFAULTS)
    if dataset_name is None:
        return defaults

    optimizer_defaults = _default_scan_optimizer_config(dataset_name)
    if str(optimizer_defaults["type"]).strip().lower() != "sgd":
        return defaults

    defaults.update(
        {
            "scan_batch_size": int(optimizer_defaults["batch"]),
            "scan_optimizer": str(optimizer_defaults["type"]),
            "scan_learning_rate": float(optimizer_defaults["lr"]),
            "scan_weight_decay": float(optimizer_defaults["wd"]),
            "scan_momentum": float(optimizer_defaults["momentum"]),
        }
    )
    return defaults


def _resolve_scan_checkpoint_fallback_token(
    openclip_pretraining: str,
    openclip_backbone: str,
) -> str | None:
    if _uses_scan_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return SCAN_GENERIC_OPENCLIP_CACHE_TOKEN
    return None


def _resolve_scan_hyperparameter_defaults(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> dict[str, Any]:
    if _uses_scan_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return _resolve_scan_generic_openclip_defaults(dataset_name)
    profile_key = (
        _normalize_scan_profile_token(openclip_pretraining),
        _normalize_scan_profile_token(openclip_backbone),
        _normalize_scan_profile_dataset(dataset_name),
    )
    return dict(SCAN_HYPERPARAMETER_DEFAULTS.get(profile_key, {}))


def _resolve_scan_hyperparameter_profile_source(
    openclip_pretraining: str,
    openclip_backbone: str,
    dataset_name: str,
) -> str | None:
    if _uses_scan_generic_openclip_defaults(openclip_pretraining, openclip_backbone):
        return None
    profile_key = (
        _normalize_scan_profile_token(openclip_pretraining),
        _normalize_scan_profile_token(openclip_backbone),
        _normalize_scan_profile_dataset(dataset_name),
    )
    if profile_key not in SCAN_HYPERPARAMETER_DEFAULTS:
        return None
    return "/".join(profile_key)


def _resolve_scan_num_neighbors(params: dict[str, Any], defaults: dict[str, Any], dataset_name: str) -> int:
    if "num_neighbors" in params:
        return int(params["num_neighbors"])
    if "knn" in params:
        return int(params["knn"])
    return int(defaults.get("num_neighbors", _default_scan_num_neighbors(dataset_name)))


def _resolve_scan_num_heads(params: dict[str, Any], defaults: dict[str, Any], dataset_name: str) -> int:
    if "num_heads" in params:
        return int(params["num_heads"])
    if "head" in params:
        return int(params["head"])
    if "heads" in params:
        return int(params["heads"])
    return int(defaults.get("num_heads", _default_scan_num_heads(dataset_name)))


def _resolve_scan_entropy_weight(params: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "entropy_weight" in params:
        return float(params["entropy_weight"])
    return float(defaults.get("entropy_weight", 5.0))


def _resolve_scan_learning_rate(params: dict[str, Any], defaults: dict[str, Any], fallback: Any) -> float:
    if "scan_learning_rate" in params:
        return float(params["scan_learning_rate"])
    if "lr" in params:
        return float(params["lr"])
    return float(defaults.get("scan_learning_rate", fallback))


class EMA:
    def __init__(self, model, alpha: float = 0.999) -> None:
        self.shadow = {key: value.clone().detach() for key, value in model.state_dict().items()}
        self.param_keys = [key for key, _ in model.named_parameters()]
        self.alpha = alpha

    def update_params(self, model) -> None:
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(self.alpha * self.shadow[name] + (1 - self.alpha) * state[name])

    def apply_shadow(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)


class _AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.sum / max(self.count, 1)


class Augment:
    def __init__(self, n: int) -> None:
        self.n = n
        self.augment_list = _augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = random.random() * float(maxval - minval) + minval
            img = op(img, val)
        return img


class Cutout:
    def __init__(self, n_holes: int, length: int, random_length: bool = False) -> None:
        self.n_holes = n_holes
        self.length = length
        self.random_length = random_length

    def __call__(self, img):
        torch = require_module("torch", "pip install torch")
        height = img.size(1)
        width = img.size(2)
        length = random.randint(1, self.length) if self.random_length else self.length
        mask = np.ones((height, width), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)
            y1 = np.clip(y - length // 2, 0, height)
            y2 = np.clip(y + length // 2, 0, height)
            x1 = np.clip(x - length // 2, 0, width)
            x2 = np.clip(x + length // 2, 0, width)
            mask[y1:y2, x1:x2] = 0.0
        mask_tensor = torch.from_numpy(mask).expand_as(img)
        return img * mask_tensor


class ScanAugmentedPairDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, dataset: LoadedImageDataset, image_transform, augmentation_transform) -> None:
        self.dataset = dataset
        self.image_transform = image_transform
        self.augmentation_transform = augmentation_transform
        transforms = require_module("torchvision.transforms", "pip install torch torchvision")
        self.to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.dataset.sample_indices)

    def _get_image(self, index: int):
        dataset_index = self.dataset.sample_indices[index]
        image, raw_label = self.dataset.dataset[dataset_index]
        if hasattr(image, "detach"):
            image = self.to_pil(image)
        label = get_image_label(self.dataset, dataset_index, raw_label)
        return image, label

    def __getitem__(self, index: int):
        image, label = self._get_image(index)
        return {
            "image": self.image_transform(image),
            "image_augmented": self.augmentation_transform(image),
            "target": label,
        }


class ScanNeighborsDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        dataset: LoadedImageDataset,
        indices: np.ndarray,
        anchor_transform,
        neighbor_transform,
        num_neighbors: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.anchor_transform = anchor_transform
        self.neighbor_transform = neighbor_transform
        self.indices = np.asarray(indices, dtype=np.int64)
        if num_neighbors is not None:
            self.indices = self.indices[:, : num_neighbors + 1]
        transforms = require_module("torchvision.transforms", "pip install torch torchvision")
        self.to_pil = transforms.ToPILImage()

    def __len__(self) -> int:
        return len(self.dataset.sample_indices)

    def _get_image(self, index: int):
        dataset_index = self.dataset.sample_indices[index]
        image, raw_label = self.dataset.dataset[dataset_index]
        if hasattr(image, "detach"):
            image = self.to_pil(image)
        label = get_image_label(self.dataset, dataset_index, raw_label)
        return image, label

    def __getitem__(self, index: int):
        anchor_image, label = self._get_image(index)
        neighbor_index = int(np.random.choice(self.indices[index], 1)[0])
        neighbor_image, _ = self._get_image(neighbor_index)
        return {
            "anchor": self.anchor_transform(anchor_image),
            "neighbor": self.neighbor_transform(neighbor_image),
            "possible_neighbors": self.indices[index].copy(),
            "target": label,
        }


class ScanEvalDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
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
        label = get_image_label(self.dataset, dataset_index, raw_label)
        return {"image": self.transform(image), "target": label}


class OpenCLIPContrastiveModel(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, backbone_model, backbone_dim: int, features_dim: int = 128) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.backbone = backbone_model
        self.backbone_dim = backbone_dim
        self.contrastive_head = torch.nn.Sequential(
            torch.nn.Linear(backbone_dim, backbone_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(backbone_dim, features_dim),
        )

    def encode_backbone(self, x):
        return self.backbone.encode_image(x).float()

    def forward(self, x):
        torch = require_module("torch", "pip install torch")
        features = self.contrastive_head(self.encode_backbone(x))
        return torch.nn.functional.normalize(features, dim=1)


class OpenCLIPClusteringModel(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, backbone_model, backbone_dim: int, nclusters: int, nheads: int = 1) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.backbone = backbone_model
        self.backbone_dim = backbone_dim
        self.nheads = nheads
        self.cluster_head = torch.nn.ModuleList(
            [torch.nn.Linear(backbone_dim, nclusters) for _ in range(nheads)]
        )

    def encode_backbone(self, x):
        return self.backbone.encode_image(x).float()

    def forward(self, x, forward_pass: str = "default"):
        if forward_pass == "default":
            features = self.encode_backbone(x)
            return [head(features) for head in self.cluster_head]
        if forward_pass == "backbone":
            return self.encode_backbone(x)
        if forward_pass == "head":
            return [head(x) for head in self.cluster_head]
        if forward_pass == "return_all":
            features = self.encode_backbone(x)
            return {"features": features, "output": [head(features) for head in self.cluster_head]}
        raise ValueError(f"Invalid forward pass {forward_pass}")


class ScanFeatureNeighborsDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        random_neighbor: bool,
        include_self: bool = True,
    ) -> None:
        torch = require_module("torch", "pip install torch")
        self.features = torch.as_tensor(features.astype("float32", copy=False))
        self.labels = torch.as_tensor(labels.astype(np.int64, copy=False))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.random_neighbor = random_neighbor
        self.include_self = include_self

    def __len__(self) -> int:
        return int(self.features.size(0))

    def _candidate_neighbors(self, index: int) -> np.ndarray:
        candidates = self.indices[index]
        if not self.include_self:
            candidates = candidates[candidates != index]
        if candidates.size == 0:
            candidates = np.asarray([index], dtype=np.int64)
        return candidates

    def __getitem__(self, index: int):
        candidates = self._candidate_neighbors(index)
        neighbor_index = int(np.random.choice(candidates, 1)[0]) if self.random_neighbor else int(candidates[0])
        return {
            "anchor_features": self.features[index],
            "neighbor_features": self.features[neighbor_index],
            "possible_neighbors": self.indices[index].copy(),
            "target": self.labels[index],
        }


class ScanFeatureEvalDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        torch = require_module("torch", "pip install torch")
        self.features = torch.as_tensor(features.astype("float32", copy=False))
        self.labels = torch.as_tensor(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.features.size(0))

    def __getitem__(self, index: int):
        return {"features": self.features[index], "target": self.labels[index]}


class ScanFeatureClusteringModel(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, feature_dim: int, nclusters: int, nheads: int = 1) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.nheads = nheads
        self.cluster_head = torch.nn.ModuleList([torch.nn.Linear(feature_dim, nclusters) for _ in range(nheads)])

    def forward(self, x):
        return [head(x.float()) for head in self.cluster_head]


def run_scan_pipeline(inputs: MethodInputs, params: dict[str, Any]) -> SCANOutputs:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("scan", params)

    device = torch.device(inputs.runtime_config.device)
    openclip_pretraining = str(params.get("openclip_pretraining", "LAION400M"))
    openclip_backbone = str(params.get("openclip_backbone", "ViT-B/32"))
    progress.log(f"Loading OpenCLIP model '{openclip_backbone}' pretrained on '{openclip_pretraining}'")
    bundle = load_openclip_bundle(openclip_pretraining, openclip_backbone, str(device))
    output_dir = method_feature_cache_dir(
        resolve_benchmark_data_root(inputs.image_dataset_config.root),
        "scan",
        bundle.spec.cache_key,
    )

    train_dataset, eval_dataset, train_split, eval_split, domain_shift = _resolve_domain_shift_datasets(
        inputs,
        params,
        method_name="SCAN",
    )
    progress.log(
        f"Resolved data protocol: train={train_dataset.name}/{train_split}, "
        f"eval={eval_dataset.name}/{eval_split}"
    )
    if train_dataset.num_classes == 0:
        raise ValueError("SCAN requires datasets with known class names on the source train split.")
    if eval_dataset.labels is None:
        raise ValueError("SCAN requires labels on the evaluation split.")
    if train_dataset.num_samples < 2:
        raise ValueError("SCAN requires at least two source-train samples to mine nearest neighbors.")
    if eval_dataset.num_samples < 2:
        raise ValueError("SCAN requires at least two evaluation samples for official SCAN-loss selection.")

    uses_generic_defaults = _uses_scan_generic_openclip_defaults(
        openclip_pretraining,
        openclip_backbone,
    )
    large_scale_head_profile = (
        False if uses_generic_defaults else _uses_large_scale_scan_heads(train_dataset.name)
    )
    scan_default_opt = _default_scan_optimizer_config(train_dataset.name)
    hyperparameter_defaults = _resolve_scan_hyperparameter_defaults(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    hyperparameter_profile_source = _resolve_scan_hyperparameter_profile_source(
        openclip_pretraining,
        openclip_backbone,
        train_dataset.name,
    )
    image_size = _resolve_openclip_image_size(bundle)
    n_clusters, n_clusters_source = resolve_source_default_cluster_num(
        params,
        train_dataset,
        method_name="SCAN",
    )
    if n_clusters <= 0:
        raise ValueError("SCAN requires n_clusters > 0.")
    num_neighbors = _resolve_scan_num_neighbors(params, hyperparameter_defaults, train_dataset.name)
    if num_neighbors <= 0:
        raise ValueError("SCAN requires num_neighbors > 0.")
    effective_num_neighbors = min(num_neighbors, train_dataset.num_samples - 1)
    if effective_num_neighbors != num_neighbors:
        progress.log(
            f"Reducing SCAN neighbors from {num_neighbors} to {effective_num_neighbors} "
            f"for {train_dataset.num_samples} source-train sample(s)"
        )
    selection_num_neighbors = int(params.get("selection_num_neighbors", 5))
    if selection_num_neighbors <= 0:
        raise ValueError("SCAN requires selection_num_neighbors > 0.")
    effective_selection_num_neighbors = min(selection_num_neighbors, eval_dataset.num_samples - 1)
    if effective_selection_num_neighbors != selection_num_neighbors:
        progress.log(
            f"Reducing SCAN selection neighbors from {selection_num_neighbors} to "
            f"{effective_selection_num_neighbors} for {eval_dataset.num_samples} evaluation sample(s)"
        )
    num_heads = _resolve_scan_num_heads(params, hyperparameter_defaults, train_dataset.name)
    if num_heads <= 0:
        raise ValueError("SCAN requires num_heads > 0.")
    entropy_weight = _resolve_scan_entropy_weight(params, hyperparameter_defaults)

    scan_epochs = int(
        params.get(
            "scan_epochs",
            hyperparameter_defaults.get("scan_epochs", _default_scan_epochs(train_dataset.name)),
        )
    )
    scan_batch_size = int(
        params.get(
            "scan_batch_size",
            hyperparameter_defaults.get("scan_batch_size", scan_default_opt["batch"]),
        )
    )
    scan_optimizer = str(
        params.get(
            "scan_optimizer",
            hyperparameter_defaults.get("scan_optimizer", scan_default_opt["type"]),
        )
    )
    scan_lr = _resolve_scan_learning_rate(params, hyperparameter_defaults, scan_default_opt["lr"])
    scan_weight_decay = float(
        params.get(
            "scan_weight_decay",
            hyperparameter_defaults.get("scan_weight_decay", scan_default_opt["wd"]),
        )
    )
    scan_momentum = float(
        params.get(
            "scan_momentum",
            hyperparameter_defaults.get("scan_momentum", scan_default_opt["momentum"]),
        )
    )
    eval_batch_size = int(params.get("eval_batch_size", max(scan_batch_size, 256)))
    selection_chunk_size = int(params.get("selection_chunk_size", 1024))
    if selection_chunk_size <= 0:
        raise ValueError("SCAN requires selection_chunk_size > 0.")
    include_self_neighbors = bool(params.get("include_self_neighbors", True))
    scan_drop_last = bool(params.get("scan_drop_last", True))
    save_checkpoint = save_checkpoint_enabled(params, train_dataset.name)

    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad = False

    scan_cache_dir = ensure_dir(
        output_dir
        / cache_key(
            image_dataset_cache_key(train_dataset),
            "head-only",
        )
    )
    neighbor_cache_path = scan_cache_dir / cache_key(
        image_dataset_cache_key(train_dataset),
        bundle.spec.cache_key,
        f"neighbors{effective_num_neighbors}",
    )
    neighbor_cache_path = neighbor_cache_path.with_suffix(".npy")
    selection_cache_dir = ensure_dir(
        output_dir
        / cache_key(
            image_dataset_cache_key(eval_dataset),
            "official-selection",
        )
    )
    selection_neighbor_cache_path = selection_cache_dir / cache_key(
        image_dataset_cache_key(eval_dataset),
        bundle.spec.cache_key,
        f"neighbors{effective_selection_num_neighbors}",
    )
    selection_neighbor_cache_path = selection_neighbor_cache_path.with_suffix(".npy")

    with progress.stage("Encoding source-train images with frozen OpenCLIP"):
        train_features, train_labels = _load_common_normalized_image_features(
            dataset=train_dataset,
            bundle=bundle,
            batch_size=eval_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )
    if train_labels is None:
        train_labels = np.zeros(train_features.shape[0], dtype=np.int64)

    with progress.stage(f"Encoding evaluation split '{eval_dataset.name}/{eval_split}' with frozen OpenCLIP"):
        eval_features, evaluation_labels = _load_common_normalized_image_features(
            dataset=eval_dataset,
            bundle=bundle,
            batch_size=eval_batch_size,
            num_workers=inputs.runtime_config.num_workers,
            device=device,
        )
    if evaluation_labels is None:
        raise ValueError("SCAN requires labels on the evaluation split.")

    checkpoint_fallback_token = _resolve_scan_checkpoint_fallback_token(
        openclip_pretraining,
        openclip_backbone,
    )
    checkpoint_stem = cache_key(
        image_dataset_cache_key(train_dataset),
        f'k{n_clusters}',
        f'heads{num_heads}',
        checkpoint_fallback_token,
    )
    checkpoint_name = f"{checkpoint_stem}__scan_final.pt"
    checkpoint_path = checkpoint_file(output_dir, checkpoint_name)
    legacy_checkpoint_stem = cache_key(
        train_dataset.name,
        train_split,
        train_dataset.name,
        "val",
        f"k{n_clusters}",
        f"heads{num_heads}",
        checkpoint_fallback_token,
    )
    legacy_checkpoint_path = checkpoint_file(
        output_dir,
        f"{legacy_checkpoint_stem}__scan_final.pt",
    )
    legacy_target_checkpoint_stem = cache_key(
        image_dataset_cache_key(train_dataset),
        image_dataset_cache_key(eval_dataset),
        f"k{n_clusters}",
        f"heads{num_heads}",
        checkpoint_fallback_token,
    )
    legacy_target_checkpoint_path = checkpoint_file(
        output_dir,
        f"{legacy_target_checkpoint_stem}__scan_final.pt",
    )
    load_checkpoint = load_checkpoint_enabled(params, train_dataset.name)
    existing_checkpoint = (
        first_existing_checkpoint([checkpoint_path, legacy_checkpoint_path, legacy_target_checkpoint_path])
        if load_checkpoint
        else None
    )
    checkpoint_loaded = False
    if load_checkpoint and existing_checkpoint is not None:
        progress.log(f"Loading SCAN checkpoint and skipping head training: {existing_checkpoint}")
        predictions, evaluation_labels, scan_history, best_scan_head, best_scan_loss = _infer_scan_checkpoint(
            eval_features=eval_features,
            evaluation_labels=evaluation_labels,
            n_clusters=n_clusters,
            num_heads=num_heads,
            selected_head=int(params.get("selected_head", -1)),
            batch_size=eval_batch_size,
            device=device,
            checkpoint_path=existing_checkpoint,
        )
        checkpoint_loaded = True
        return SCANOutputs(
            predictions=predictions.tolist(),
            evaluation_labels=evaluation_labels.tolist(),
            evaluation_split=eval_split,
            metadata={
                "variant": "scan_head_only",
                "source_dataset": train_dataset.name,
                "evaluation_dataset": eval_dataset.name,
                "domain_shift": domain_shift,
                "train_split": train_split,
                "test_split": eval_split,
                "openclip_pretraining": bundle.spec.benchmark_pretraining,
                "openclip_backbone": bundle.spec.benchmark_backbone,
                "n_clusters": n_clusters,
                "n_clusters_source": n_clusters_source,
                "num_heads": num_heads,
                "selected_scan_head": best_scan_head,
                "selected_scan_loss": best_scan_loss,
                "save_checkpoint": save_checkpoint,
                "checkpoint_saved": False,
                "checkpoint_path": str(existing_checkpoint),
                "load_checkpoint": load_checkpoint,
                "checkpoint_loaded": checkpoint_loaded,
                "scan_history": scan_history,
            },
        )

    with progress.stage(f"Mining SCAN train nearest neighbors with topk={effective_num_neighbors}"):
        neighbor_indices = _mine_neighbors_from_features(
            train_features,
            topk=effective_num_neighbors,
            cache_path=neighbor_cache_path,
            progress=progress,
        )
    with progress.stage(
        f"Mining SCAN official-selection nearest neighbors with topk={effective_selection_num_neighbors}"
    ):
        selection_neighbor_indices = _mine_neighbors_from_features(
            eval_features,
            topk=effective_selection_num_neighbors,
            cache_path=selection_neighbor_cache_path,
            progress=progress,
        )

    scan_model = ScanFeatureClusteringModel(
        feature_dim=int(train_features.shape[1]),
        nclusters=n_clusters,
        nheads=num_heads,
    ).to(device)
    scan_optimizer_obj = _build_optimizer(
        optimizer_name=scan_optimizer,
        model=scan_model,
        learning_rate=scan_lr,
        weight_decay=scan_weight_decay,
        momentum=scan_momentum,
        cluster_head_only=True,
    )
    scan_criterion = SCANLoss(entropy_weight=entropy_weight).to(device)
    scan_train_loader = data_mod.DataLoader(
        ScanFeatureNeighborsDataset(
            train_features,
            train_labels,
            neighbor_indices,
            random_neighbor=True,
            include_self=include_self_neighbors,
        ),
        batch_size=scan_batch_size,
        shuffle=True,
        drop_last=scan_drop_last,
        num_workers=0,
    )
    eval_loader = data_mod.DataLoader(ScanFeatureEvalDataset(eval_features, evaluation_labels), batch_size=eval_batch_size)

    best_scan_loss = float("inf")
    best_scan_head = 0
    best_scan_state = copy.deepcopy(scan_model.state_dict())
    scan_history: list[dict[str, Any]] = []
    efficiency_measurement = start_train_eval_measurement(device)
    progress.log(
        f"Training frozen-backbone SCAN clustering head module for {scan_epochs} epoch(s) "
        f"with {num_heads} head(s)"
    )
    for epoch in range(scan_epochs):
        lr = _adjust_learning_rate(
            optimizer=scan_optimizer_obj,
            base_lr=scan_lr,
            epoch=epoch,
            epochs=scan_epochs,
            scheduler="constant",
            lr_decay_rate=0.1,
            lr_decay_epochs=None,
        )
        train_stats = _scan_feature_train_epoch(
            scan_train_loader,
            scan_model,
            scan_criterion,
            scan_optimizer_obj,
            device,
        )
        eval_stats = _evaluate_feature_scan_loss_official(
            eval_loader,
            scan_model,
            selection_neighbor_indices,
            device,
            chunk_size=selection_chunk_size,
        )
        epoch_record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_stats["total_loss"],
            "train_consistency": train_stats["consistency"],
            "train_entropy": train_stats["entropy"],
            "selection_lowest_loss_head": eval_stats["lowest_loss_head"],
            "selection_lowest_loss": eval_stats["lowest_loss"],
        }
        scan_history.append(epoch_record)
        progress.epoch(
            "SCAN training",
            epoch + 1,
            scan_epochs,
            metrics={
                "train_loss": epoch_record["train_loss"],
                "select_loss": epoch_record["selection_lowest_loss"],
            },
        )
        if eval_stats["lowest_loss"] < best_scan_loss:
            best_scan_loss = float(eval_stats["lowest_loss"])
            best_scan_head = int(eval_stats["lowest_loss_head"])
            best_scan_state = copy.deepcopy(scan_model.state_dict())

    scan_model.load_state_dict(best_scan_state, strict=True)
    if save_checkpoint:
        checkpoint_path = checkpoint_file(output_dir, checkpoint_name, create=True)
        torch.save(
            {
                "model": scan_model.state_dict(),
                "optimizer": scan_optimizer_obj.state_dict(),
                "n_clusters": n_clusters,
                "num_heads": num_heads,
                "selected_head": best_scan_head,
                "selected_scan_loss": best_scan_loss,
                "epochs": scan_epochs,
                "history": scan_history,
            },
            checkpoint_path,
        )

    with progress.stage(f"Predicting evaluation split '{eval_dataset.name}/{eval_split}'"):
        prediction_heads = _get_feature_predictions(eval_loader, scan_model, device)
    selected_predictions = prediction_heads[best_scan_head]
    predictions = selected_predictions["predictions"].numpy().astype(np.int64)
    evaluation_labels = selected_predictions["targets"].numpy().astype(np.int64)
    efficiency_measurement.stop()

    return SCANOutputs(
        predictions=predictions.tolist(),
        evaluation_labels=evaluation_labels.tolist(),
        evaluation_split=eval_split,
        metadata={
            "variant": "scan_head_only",
            "source_dataset": train_dataset.name,
            "evaluation_dataset": eval_dataset.name,
            "domain_shift": domain_shift,
            "train_split": train_split,
            "test_split": eval_split,
            "openclip_pretraining": bundle.spec.benchmark_pretraining,
            "openclip_backbone": bundle.spec.benchmark_backbone,
            "n_clusters": n_clusters,
            "n_clusters_source": n_clusters_source,
            "image_size": image_size,
            "backbone_frozen": True,
            "run_pretext": False,
            "run_selflabel": False,
            "feature_dim": int(train_features.shape[1]),
            "scan_batch_size": scan_batch_size,
            "eval_batch_size": eval_batch_size,
            "num_neighbors": effective_num_neighbors,
            "requested_num_neighbors": num_neighbors,
            "selection_num_neighbors": effective_selection_num_neighbors,
            "requested_selection_num_neighbors": selection_num_neighbors,
            "num_heads": num_heads,
            "default_num_heads": int(
                hyperparameter_defaults.get("num_heads", _default_scan_num_heads(train_dataset.name))
            ),
            "default_scan_profile": (
                "fallback" if uses_generic_defaults else _default_scan_profile_name(train_dataset.name)
            ),
            "hyperparameter_default_profile": hyperparameter_profile_source,
            "large_scale_head_profile": large_scale_head_profile,
            "update_cluster_head_only": True,
            "include_self_neighbors": include_self_neighbors,
            "scan_drop_last": scan_drop_last,
            "entropy_weight": entropy_weight,
            "selection_entropy_weight": 1.0,
            "selection_split": eval_split,
            "selection_dataset": eval_dataset.name,
            "selection_protocol": "official_scan_eval_loss",
            "selection_chunk_size": selection_chunk_size,
            "scan_optimizer": scan_optimizer,
            "scan_learning_rate": scan_lr,
            "scan_weight_decay": scan_weight_decay,
            "scan_momentum": scan_momentum,
            "neighbor_feature_dim": int(train_features.shape[1]),
            "selected_scan_head": best_scan_head,
            "selected_scan_loss": best_scan_loss,
            "scan_cache_dir": str(scan_cache_dir),
            "neighbor_cache_path": str(neighbor_cache_path),
            "selection_neighbor_cache_path": str(selection_neighbor_cache_path),
            "shared_feature_cache": "common raw OpenCLIP image features",
            "save_checkpoint": save_checkpoint,
            "checkpoint_saved": save_checkpoint,
            "checkpoint_path": str(checkpoint_path) if (checkpoint_loaded or save_checkpoint) else None,
            "load_checkpoint": load_checkpoint,
            "checkpoint_loaded": checkpoint_loaded,
            "scan_history": scan_history,
            "protocol_note": (
                "This benchmark version freezes the OpenCLIP backbone, trains only the SCAN clustering head module "
                "on source-train nearest-neighbor consistency plus entropy regularization, does not use image "
                "augmentation, and follows the official SCAN validation-side SCAN-loss protocol to select the best "
                "head/checkpoint state."
            ),
        },
    )


def _infer_scan_checkpoint(
    *,
    eval_features: np.ndarray,
    evaluation_labels: np.ndarray,
    n_clusters: int,
    num_heads: int,
    selected_head: int,
    batch_size: int,
    device,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int, float | None]:
    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    payload = load_torch_checkpoint(checkpoint_path, device=device)
    scan_model = ScanFeatureClusteringModel(
        feature_dim=int(eval_features.shape[1]),
        nclusters=n_clusters,
        nheads=num_heads,
    ).to(device)
    scan_model.load_state_dict(payload["model"], strict=True)
    best_head = int(payload.get("selected_head", payload.get("best_head", 0)))
    if selected_head >= 0:
        best_head = selected_head
    eval_loader = data_mod.DataLoader(ScanFeatureEvalDataset(eval_features, evaluation_labels), batch_size=batch_size)
    prediction_heads = _get_feature_predictions(eval_loader, scan_model, device)
    selected_predictions = prediction_heads[best_head]
    predictions = selected_predictions["predictions"].numpy().astype(np.int64)
    labels = selected_predictions["targets"].numpy().astype(np.int64)
    history = payload.get("history", [])
    selected_loss = payload.get("selected_scan_loss")
    return predictions, labels, history if isinstance(history, list) else [], best_head, selected_loss


def _build_train_transform(
    *,
    strategy: str,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    simclr_scale,
    num_strong_augs: int,
    cutout_holes: int,
    cutout_length: int,
    cutout_random: bool,
):
    transforms = require_module("torchvision.transforms", "pip install torch torchvision")
    strategy_key = strategy.strip().lower()
    if strategy_key == "simclr":
        scale = tuple(float(value) for value in simclr_scale)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=image_size, scale=scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    if strategy_key == "ours":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=max(4, image_size // 8), padding_mode="reflect"),
                Augment(num_strong_augs),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                Cutout(
                    n_holes=cutout_holes,
                    length=cutout_length,
                    random_length=cutout_random,
                ),
            ]
        )
    raise ValueError(f"Invalid augmentation strategy '{strategy}' for SCAN.")


def _build_val_transform(*, image_size: int, mean: tuple[float, float, float], std: tuple[float, float, float]):
    transforms = require_module("torchvision.transforms", "pip install torch torchvision")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _resolve_openclip_rep_dim(bundle: OpenCLIPBundle) -> int:
    rep_dim = getattr(bundle.model.visual, "output_dim", None)
    if rep_dim is not None:
        return int(rep_dim)
    text_projection = getattr(bundle.model, "text_projection", None)
    if text_projection is not None:
        return int(text_projection.shape[1])
    raise ValueError("Unable to infer OpenCLIP visual representation dimension for SCAN.")


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
            mean = tuple(float(value) for value in transform.mean)
            std = tuple(float(value) for value in transform.std)
            return mean, std
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def _build_optimizer(
    *,
    optimizer_name: str,
    model,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    cluster_head_only: bool,
):
    torch = require_module("torch", "pip install torch")
    if cluster_head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "cluster_head" in name
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        params = model.parameters()

    key = optimizer_name.strip().lower()
    if key == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=False,
        )
    if key == "adam":
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Invalid optimizer '{optimizer_name}' for SCAN.")


def _adjust_learning_rate(
    *,
    optimizer,
    base_lr: float,
    epoch: int,
    epochs: int,
    scheduler: str,
    lr_decay_rate: float,
    lr_decay_epochs,
) -> float:
    if scheduler == "cosine":
        eta_min = base_lr * (lr_decay_rate**3)
        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / epochs)) / 2
    elif scheduler == "step":
        if lr_decay_epochs is None:
            raise ValueError("step scheduler requires lr_decay_epochs")
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        lr = base_lr * (lr_decay_rate**steps)
    elif scheduler == "constant":
        lr = base_lr
    else:
        raise ValueError(f"Invalid scheduler '{scheduler}' for SCAN.")
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def _simclr_train_epoch(train_loader, model, criterion, optimizer, device) -> float:
    torch = require_module("torch", "pip install torch")
    model.train()
    loss_meter = _AverageMeter()
    for batch in train_loader:
        images = batch["image"]
        images_augmented = batch["image_augmented"]
        batch_size, channels, height, width = images.size()
        stacked = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1).view(-1, channels, height, width)
        stacked = stacked.to(device, non_blocking=True)
        output = model(stacked).view(batch_size, 2, -1)
        loss = criterion(output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(float(loss.item()), batch_size)
    return loss_meter.average


def _scan_train_epoch(train_loader, model, criterion, optimizer, device, *, update_cluster_head_only: bool) -> dict[str, float]:
    torch = require_module("torch", "pip install torch")
    total_meter = _AverageMeter()
    consistency_meter = _AverageMeter()
    entropy_meter = _AverageMeter()
    model.eval() if update_cluster_head_only else model.train()

    for batch in train_loader:
        anchors = batch["anchor"].to(device, non_blocking=True)
        neighbors = batch["neighbor"].to(device, non_blocking=True)
        if update_cluster_head_only:
            with torch.no_grad():
                anchor_features = model(anchors, forward_pass="backbone")
                neighbor_features = model(neighbors, forward_pass="backbone")
            anchor_outputs = model(anchor_features, forward_pass="head")
            neighbor_outputs = model(neighbor_features, forward_pass="head")
        else:
            anchor_outputs = model(anchors)
            neighbor_outputs = model(neighbors)

        total_losses = []
        consistency_losses = []
        entropy_losses = []
        for anchor_output, neighbor_output in zip(anchor_outputs, neighbor_outputs):
            total_loss, consistency_loss, entropy_loss = criterion(anchor_output, neighbor_output)
            total_losses.append(total_loss)
            consistency_losses.append(consistency_loss)
            entropy_losses.append(entropy_loss)

        total_value = torch.sum(torch.stack(total_losses, dim=0))
        optimizer.zero_grad()
        total_value.backward()
        optimizer.step()

        batch_size = anchors.size(0)
        total_meter.update(float(np.mean([value.item() for value in total_losses])), batch_size)
        consistency_meter.update(float(np.mean([value.item() for value in consistency_losses])), batch_size)
        entropy_meter.update(float(np.mean([value.item() for value in entropy_losses])), batch_size)

    return {
        "total_loss": total_meter.average,
        "consistency": consistency_meter.average,
        "entropy": entropy_meter.average,
    }


def _scan_feature_train_epoch(train_loader, model, criterion, optimizer, device) -> dict[str, float]:
    torch = require_module("torch", "pip install torch")
    total_meter = _AverageMeter()
    consistency_meter = _AverageMeter()
    entropy_meter = _AverageMeter()
    model.train()

    for batch in train_loader:
        anchors = batch["anchor_features"].to(device, non_blocking=True)
        neighbors = batch["neighbor_features"].to(device, non_blocking=True)
        anchor_outputs = model(anchors)
        neighbor_outputs = model(neighbors)

        total_losses = []
        consistency_losses = []
        entropy_losses = []
        for anchor_output, neighbor_output in zip(anchor_outputs, neighbor_outputs):
            total_loss, consistency_loss, entropy_loss = criterion(anchor_output, neighbor_output)
            total_losses.append(total_loss)
            consistency_losses.append(consistency_loss)
            entropy_losses.append(entropy_loss)

        total_value = torch.sum(torch.stack(total_losses, dim=0))
        optimizer.zero_grad()
        total_value.backward()
        optimizer.step()

        batch_size = anchors.size(0)
        total_meter.update(float(np.mean([value.item() for value in total_losses])), batch_size)
        consistency_meter.update(float(np.mean([value.item() for value in consistency_losses])), batch_size)
        entropy_meter.update(float(np.mean([value.item() for value in entropy_losses])), batch_size)

    return {
        "total_loss": total_meter.average,
        "consistency": consistency_meter.average,
        "entropy": entropy_meter.average,
    }


@require_module("torch", "pip install torch").no_grad()
def _evaluate_feature_scan_loss(dataloader, model, criterion, device) -> dict[str, Any]:
    model.eval()
    total_meters = [_AverageMeter() for _ in range(model.nheads)]
    consistency_meters = [_AverageMeter() for _ in range(model.nheads)]
    entropy_meters = [_AverageMeter() for _ in range(model.nheads)]

    for batch in dataloader:
        anchors = batch["anchor_features"].to(device, non_blocking=True)
        neighbors = batch["neighbor_features"].to(device, non_blocking=True)
        anchor_outputs = model(anchors)
        neighbor_outputs = model(neighbors)

        batch_size = anchors.size(0)
        for head_index, (anchor_output, neighbor_output) in enumerate(zip(anchor_outputs, neighbor_outputs)):
            total_loss, consistency_loss, entropy_loss = criterion(anchor_output, neighbor_output)
            total_meters[head_index].update(float(total_loss.item()), batch_size)
            consistency_meters[head_index].update(float(consistency_loss.item()), batch_size)
            entropy_meters[head_index].update(float(entropy_loss.item()), batch_size)

    output = [
        {
            "entropy": entropy_meter.average,
            "consistency": consistency_meter.average,
            "total_loss": total_meter.average,
        }
        for total_meter, consistency_meter, entropy_meter in zip(total_meters, consistency_meters, entropy_meters)
    ]
    total_losses = [item["total_loss"] for item in output]
    return {
        "scan": output,
        "lowest_loss_head": int(np.argmin(total_losses)),
        "lowest_loss": float(np.min(total_losses)),
    }


@require_module("torch", "pip install torch").no_grad()
def _evaluate_feature_scan_loss_official(
    dataloader,
    model,
    neighbor_indices: np.ndarray,
    device,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    torch = require_module("torch", "pip install torch")
    functional = require_module("torch.nn.functional", "pip install torch")
    predictions = _get_feature_predictions(dataloader, model, device)
    neighbors = torch.as_tensor(neighbor_indices.astype(np.int64, copy=False), dtype=torch.long, device=device)
    output = []

    for head in predictions:
        probs = head["probabilities"].to(device)
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        consistency_sum = 0.0
        consistency_count = 0
        for start in range(0, probs.size(0), chunk_size):
            end = min(start + chunk_size, probs.size(0))
            similarity = torch.matmul(probs[start:end], probs.t())
            selected = similarity.gather(1, neighbors[start:end])
            ones = torch.ones_like(selected)
            consistency_sum += float(functional.binary_cross_entropy(selected, ones, reduction="sum").item())
            consistency_count += int(selected.numel())

        consistency_loss = consistency_sum / max(consistency_count, 1)
        total_loss = consistency_loss - entropy_loss
        output.append(
            {
                "entropy": entropy_loss,
                "consistency": consistency_loss,
                "total_loss": total_loss,
            }
        )

    total_losses = [item["total_loss"] for item in output]
    return {
        "scan": output,
        "lowest_loss_head": int(np.argmin(total_losses)),
        "lowest_loss": float(np.min(total_losses)),
    }


def _selflabel_train_epoch(train_loader, model, criterion, optimizer, device, *, ema: EMA | None) -> float:
    model.train()
    loss_meter = _AverageMeter()
    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        images_augmented = batch["image_augmented"].to(device, non_blocking=True)
        with require_module("torch", "pip install torch").no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]
        loss = criterion(output, output_augmented)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update_params(model)
            ema.apply_shadow(model)
        loss_meter.update(float(loss.item()), images.size(0))
    return loss_meter.average


@require_module("torch", "pip install torch").no_grad()
def _extract_features(dataloader, *, feature_model, backbone_model, device) -> tuple[np.ndarray, np.ndarray]:
    torch = require_module("torch", "pip install torch")
    features = []
    targets = []
    if feature_model is not None:
        feature_model.eval()
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        if feature_model is not None:
            output = feature_model(images)
        else:
            output = backbone_model.encode_image(images).float()
            output = torch.nn.functional.normalize(output, dim=1)
        features.append(output.cpu().numpy().astype("float32"))
        targets.append(batch["target"].numpy().astype(np.int64))
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def _load_common_normalized_image_features(
    *,
    dataset: LoadedImageDataset,
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
    normalized = raw_features / np.clip(np.linalg.norm(raw_features, axis=1, keepdims=True), 1e-12, None)
    labels_array = labels.astype(np.int64, copy=False) if labels is not None else None
    return normalized.astype("float32", copy=False), labels_array


def _mine_neighbors_from_features(
    features: np.ndarray,
    *,
    topk: int,
    cache_path: Path,
    progress=None,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == features.shape[0] and cached.shape[1] >= topk + 1:
            return cached[:, : topk + 1].astype(np.int64, copy=False)
        if progress is not None:
            progress.log(
                f"Ignoring cached SCAN neighbors with shape {cached.shape}; "
                f"expected ({features.shape[0]}, at least {topk + 1})."
            )
    indices = _search_topk_including_self(
        features,
        topk=topk,
        progress=progress,
        label="SCAN neighbor search",
    )
    np.save(cache_path, indices.astype(np.int64))
    return indices.astype(np.int64)


def _mine_neighbors(
    *,
    dataset: LoadedImageDataset,
    feature_model,
    backbone_model,
    transform,
    device,
    batch_size: int,
    num_workers: int,
    topk: int,
    cache_path: Path,
    feature_cache_path: Path,
    progress=None,
) -> tuple[np.ndarray, int]:
    if cache_path.exists() and feature_cache_path.exists():
        features = np.load(feature_cache_path)
        return np.load(cache_path), int(features.shape[1])

    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    dataloader = data_mod.DataLoader(
        ScanEvalDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    features, _ = _extract_features(
        dataloader,
        feature_model=feature_model,
        backbone_model=backbone_model,
        device=device,
    )
    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
    indices = _search_topk_including_self(
        normalized,
        topk=topk,
        progress=progress,
        label="SCAN neighbor search",
    )
    np.save(feature_cache_path, normalized.astype("float32"))
    np.save(cache_path, indices.astype(np.int64))
    return indices.astype(np.int64), int(normalized.shape[1])


def _search_topk_including_self(
    features: np.ndarray,
    *,
    topk: int,
    progress=None,
    label: str | None = None,
) -> np.ndarray:
    _, indices = search_topk(
        features,
        topk=topk + 1,
        faiss_metric="ip",
        sklearn_metric="cosine",
        progress=progress,
        label=label,
    )
    return indices.astype(np.int64)


@require_module("torch", "pip install torch").no_grad()
def _get_predictions(dataloader, model, device):
    torch = require_module("torch", "pip install torch")
    model.eval()
    predictions = [[] for _ in range(model.nheads)]
    probs = [[] for _ in range(model.nheads)]
    targets = []
    neighbors = []
    include_neighbors = isinstance(dataloader.dataset, ScanNeighborsDataset)

    for batch in dataloader:
        key = "anchor" if include_neighbors else "image"
        images = batch[key].to(device, non_blocking=True)
        output = model(images)
        for index, output_i in enumerate(output):
            predictions[index].append(torch.argmax(output_i, dim=1).cpu())
            probs[index].append(torch.nn.functional.softmax(output_i, dim=1).cpu())
        targets.append(require_module("torch", "pip install torch").as_tensor(batch["target"]))
        if include_neighbors:
            neighbors.append(require_module("torch", "pip install torch").as_tensor(batch["possible_neighbors"]))

    predictions = [torch.cat(prediction, dim=0) for prediction in predictions]
    probs = [torch.cat(prob, dim=0) for prob in probs]
    targets = torch.cat(targets, dim=0)
    if include_neighbors:
        neighbors_tensor = torch.cat(neighbors, dim=0)
        return [
            {
                "predictions": pred,
                "probabilities": prob,
                "targets": targets,
                "neighbors": neighbors_tensor,
            }
            for pred, prob in zip(predictions, probs)
        ]
    return [
        {
            "predictions": pred,
            "probabilities": prob,
            "targets": targets,
        }
        for pred, prob in zip(predictions, probs)
    ]


@require_module("torch", "pip install torch").no_grad()
def _get_feature_predictions(dataloader, model, device):
    torch = require_module("torch", "pip install torch")
    model.eval()
    predictions = [[] for _ in range(model.nheads)]
    probs = [[] for _ in range(model.nheads)]
    targets = []

    for batch in dataloader:
        features = batch["features"].to(device, non_blocking=True)
        output = model(features)
        for index, output_i in enumerate(output):
            predictions[index].append(torch.argmax(output_i, dim=1).cpu())
            probs[index].append(torch.nn.functional.softmax(output_i, dim=1).cpu())
        targets.append(torch.as_tensor(batch["target"]))

    predictions = [torch.cat(prediction, dim=0) for prediction in predictions]
    probs = [torch.cat(prob, dim=0) for prob in probs]
    targets = torch.cat(targets, dim=0)
    return [
        {
            "predictions": pred,
            "probabilities": prob,
            "targets": targets,
        }
        for pred, prob in zip(predictions, probs)
    ]


def _evaluate_scan_loss(dataloader, model, device) -> dict[str, Any]:
    predictions = _get_predictions(dataloader, model, device)
    output = []
    for head in predictions:
        probs = head["probabilities"]
        neighbors = head["neighbors"]
        anchors = require_module("torch", "pip install torch").arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)
        entropy_loss = entropy(require_module("torch", "pip install torch").mean(probs, dim=0), input_as_probabilities=True).item()
        similarity = require_module("torch", "pip install torch").matmul(probs, probs.t())
        flat_neighbors = neighbors.contiguous().view(-1)
        flat_anchors = anchors.contiguous().view(-1)
        similarity = similarity[flat_anchors, flat_neighbors]
        ones = require_module("torch", "pip install torch").ones_like(similarity)
        consistency_loss = require_module("torch.nn.functional", "pip install torch").binary_cross_entropy(similarity, ones).item()
        total_loss = -entropy_loss + consistency_loss
        output.append(
            {
                "entropy": entropy_loss,
                "consistency": consistency_loss,
                "total_loss": total_loss,
            }
        )
    total_losses = [item["total_loss"] for item in output]
    return {
        "scan": output,
        "lowest_loss_head": int(np.argmin(total_losses)),
        "lowest_loss": float(np.min(total_losses)),
    }


def _load_best_scan_head(model, scan_state: dict[str, Any], best_head: int) -> None:
    model_state = copy.deepcopy(scan_state)
    all_heads = [key for key in model_state.keys() if "cluster_head" in key]
    best_weight = model_state[f"cluster_head.{best_head}.weight"]
    best_bias = model_state[f"cluster_head.{best_head}.bias"]
    for key in all_heads:
        model_state.pop(key)
    model_state["cluster_head.0.weight"] = best_weight
    model_state["cluster_head.0.bias"] = best_bias
    model.load_state_dict(model_state, strict=True)


def _augment_list():
    pil = require_module("PIL.ImageOps", "pip install pillow")
    image_enhance = require_module("PIL.ImageEnhance", "pip install pillow")

    def shear_x(img, value):
        return img.transform(img.size, require_module("PIL.Image", "pip install pillow").AFFINE, (1, value, 0, 0, 1, 0))

    def shear_y(img, value):
        return img.transform(img.size, require_module("PIL.Image", "pip install pillow").AFFINE, (1, 0, 0, value, 1, 0))

    def identity(img, _):
        return img

    def translate_x(img, value):
        value = value * img.size[0]
        return img.transform(img.size, require_module("PIL.Image", "pip install pillow").AFFINE, (1, 0, value, 0, 1, 0))

    def translate_y(img, value):
        value = value * img.size[1]
        return img.transform(img.size, require_module("PIL.Image", "pip install pillow").AFFINE, (1, 0, 0, 0, 1, value))

    def rotate(img, value):
        return img.rotate(value)

    def auto_contrast(img, _):
        return pil.autocontrast(img)

    def equalize(img, _):
        return pil.equalize(img)

    def solarize(img, value):
        return pil.solarize(img, value)

    def posterize(img, value):
        return pil.posterize(img, int(value))

    def color(img, value):
        return image_enhance.Color(img).enhance(value)

    def contrast(img, value):
        return image_enhance.Contrast(img).enhance(value)

    def brightness(img, value):
        return image_enhance.Brightness(img).enhance(value)

    def sharpness(img, value):
        return image_enhance.Sharpness(img).enhance(value)

    return [
        (identity, 0, 1),
        (auto_contrast, 0, 1),
        (equalize, 0, 1),
        (rotate, -30, 30),
        (solarize, 0, 256),
        (color, 0.05, 0.95),
        (contrast, 0.05, 0.95),
        (brightness, 0.05, 0.95),
        (sharpness, 0.05, 0.95),
        (shear_x, -0.1, 0.1),
        (translate_x, -0.1, 0.1),
        (translate_y, -0.1, 0.1),
        (posterize, 4, 8),
        (shear_y, -0.1, 0.1),
    ]
