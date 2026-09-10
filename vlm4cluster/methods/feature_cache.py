from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label, get_image_labels_for_sample_indices
from vlm4cluster.models import OpenCLIPBundle
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.io import ensure_dir
from vlm4cluster.utils.progress import get_progress_logger


COMMON_WORDNET_PROMPT_TEMPLATE_KEY = "simple_imagenet_7prompt_v1"
COMMON_WORDNET_PROMPT_BUILDERS: tuple[Callable[[str], str], ...] = (
    lambda noun: f"itap of a {noun}.",
    lambda noun: f"a bad photo of the {noun}.",
    lambda noun: f"a origami {noun}.",
    lambda noun: f"a photo of the large {noun}.",
    lambda noun: f"a {noun} in a video game.",
    lambda noun: f"art of the {noun}.",
    lambda noun: f"a photo of the small {noun}.",
)

SIC_WORDNET_PROMPT_TEMPLATE_KEY = "sic_single_photo_v1"
SIC_WORDNET_PROMPT_BUILDERS: tuple[Callable[[str], str], ...] = (
    lambda noun: f"a photo of a {noun}",
)


class OpenCLIPImageDatasetView(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
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
        return image, label, f"image-{image_dataset_cache_key(self.dataset)}-{dataset_index}"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def cache_token(value) -> str:
    token = str(value).strip().lower()
    for old, new in (
        ("/", "-"),
        ("\\", "-"),
        (" ", "-"),
        (":", "-"),
        ("=", "-"),
        (".", "p"),
    ):
        token = token.replace(old, new)
    return "-".join(part for part in token.split("-") if part)


def cache_key(*parts) -> str:
    tokens = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text:
            continue
        tokens.append(cache_token(text))
    return "__".join(tokens)


def cache_file_path(cache_dir: Path, stem: str, suffix: str, *, max_stem_length: int = 160) -> Path:
    safe_stem = cache_key(stem)
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    filename = f"{safe_stem}{suffix}"
    if len(filename.encode("utf-8")) <= 240:
        return cache_dir / filename

    digest = hashlib.sha1(safe_stem.encode("utf-8")).hexdigest()[:16]
    prefix = safe_stem[:max_stem_length].rstrip("_-")
    return cache_dir / f"{prefix}__hash{digest}{suffix}"


def resolve_benchmark_data_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _infer_data_root_from_image_dataset(dataset: LoadedImageDataset) -> Path:
    metadata_root = dataset.metadata.get("benchmark_data_root")
    if metadata_root is not None:
        return Path(str(metadata_root)).expanduser().resolve()
    dataset_root = Path(dataset.root).expanduser().resolve()
    if dataset_root.name.lower() == dataset.split.lower():
        return dataset_root.parent.parent
    return dataset_root.parent


def wordnet_source_token(csv_path: Path) -> str:
    resolved = csv_path.expanduser().resolve()
    stats = resolved.stat()
    fingerprint = hashlib.sha1(
        f"{resolved}::{stats.st_size}::{int(stats.st_mtime)}".encode("utf-8")
    ).hexdigest()[:12]
    stem = resolved.stem.lower().replace(" ", "_")
    return f"{stem}__{fingerprint}"


def image_dataset_cache_key(dataset: LoadedImageDataset) -> str:
    attack = dataset.metadata.get("attack")
    if isinstance(attack, dict) and attack.get("cache_token"):
        return cache_key(dataset.name, dataset.split, attack["cache_token"])
    return cache_key(dataset.name, dataset.split)


def common_feature_cache_dir(data_root: str | Path, modality: str) -> Path:
    return ensure_dir(resolve_benchmark_data_root(data_root) / "feature_cache" / "common" / modality)


def method_feature_cache_dir(data_root: str | Path, method_name: str, model_cache_key: str) -> Path:
    method_token = method_name.strip().lower().replace(" ", "_")
    return ensure_dir(
        resolve_benchmark_data_root(data_root)
        / "feature_cache"
        / "method_specific"
        / method_token
        / model_cache_key
    )


def raw_image_embedding_cache_path(dataset: LoadedImageDataset, model_cache_key: str) -> Path:
    cache_dir = common_feature_cache_dir(_infer_data_root_from_image_dataset(dataset), "image")
    dataset_key = image_dataset_cache_key(dataset)
    return cache_file_path(cache_dir, f"{dataset_key}__{model_cache_key}__raw_image", ".npz")


def load_or_compute_raw_image_embeddings(
    dataset: LoadedImageDataset,
    bundle: OpenCLIPBundle,
    batch_size: int,
    num_workers: int,
    device,
) -> tuple[np.ndarray, np.ndarray | None]:
    progress = get_progress_logger("features")
    cache_path = raw_image_embedding_cache_path(dataset, bundle.spec.cache_key)
    if cache_path.exists():
        progress.log(f"Loading cached image embeddings for {dataset.name}/{dataset.split}: {cache_path}")
        payload = np.load(cache_path, allow_pickle=True)
        features = payload["features"]
        cached_sample_indices = payload["sample_indices"] if "sample_indices" in payload.files else None
        sample_order_matches = cached_sample_indices is None or np.array_equal(
            cached_sample_indices,
            np.asarray(dataset.sample_indices, dtype=np.int64),
        )
        if features.shape[0] == len(dataset.sample_indices) and sample_order_matches:
            labels = get_image_labels_for_sample_indices(dataset)
            if labels is None and bool(payload["has_labels"][0]):
                labels = payload["labels"].tolist()
            return features, np.asarray(labels, dtype=np.int64) if labels is not None else None
        reason = (
            f"{features.shape[0]} cached row(s) versus {len(dataset.sample_indices)} current sample(s)"
            if features.shape[0] != len(dataset.sample_indices)
            else "cached sample order differs from the current dataset view"
        )
        progress.log(f"Ignoring cached image embeddings because {reason}.")

    torch = require_module("torch", "pip install torch")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    view = OpenCLIPImageDatasetView(dataset, bundle.preprocess)
    loader = data_mod.DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    progress.log(
        f"Encoding {len(view)} images for {dataset.name}/{dataset.split} "
        f"with {bundle.spec.benchmark_backbone} ({len(loader)} batches)"
    )

    features = []
    labels = []
    for batch_index, (images, batch_labels, _) in enumerate(loader, start=1):
        images = images.to(device)
        with torch.no_grad():
            encoded = bundle.model.encode_image(images)
        features.append(encoded.float().cpu().numpy().astype("float32"))
        labels.extend(int(label) for label in batch_labels.tolist())
        progress.step("Image encoding", batch_index, len(loader), noun="batch")

    feature_matrix = np.concatenate(features, axis=0)
    label_array = np.asarray(labels, dtype=np.int64) if dataset.labels is not None else None
    np.savez_compressed(
        cache_path,
        features=feature_matrix,
        labels=np.asarray(labels, dtype=np.int64) if label_array is not None else np.asarray([], dtype=np.int64),
        has_labels=np.asarray([label_array is not None], dtype=bool),
        sample_indices=np.asarray(dataset.sample_indices, dtype=np.int64),
    )
    progress.log(f"Saved image embeddings cache: {cache_path}")
    return feature_matrix, label_array


def load_or_compute_text_prompt_bank(
    *,
    csv_path: Path,
    nouns: list[str],
    bundle: OpenCLIPBundle,
    batch_size: int,
    device,
    cache_dir: Path,
    prompt_builders: Sequence[Callable[[str], str]],
    template_key: str,
    normalize_prompt_embeddings: bool,
    normalize_ensemble: bool,
    force_recompute: bool,
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    progress = get_progress_logger("features")
    cache_path = cache_file_path(
        ensure_dir(cache_dir),
        f"wordnet_nouns__{wordnet_source_token(csv_path)}__{bundle.spec.cache_key}__{template_key}",
        ".npz",
    )
    if cache_path.exists() and not force_recompute:
        progress.log(f"Loading cached text prompt bank: {cache_path}")
        payload = np.load(cache_path, allow_pickle=True)
        cached_nouns = [str(noun) for noun in payload["nouns"].tolist()]
        stacked = payload["prompt_embeddings"]
        return cached_nouns, [stacked[index] for index in range(stacked.shape[0])], payload["ensemble_embeddings"]

    torch = require_module("torch", "pip install torch")
    progress.log(
        f"Encoding {len(nouns)} nouns with {len(prompt_builders)} prompt template(s) "
        f"using {bundle.spec.benchmark_backbone}"
    )
    prompt_embeddings: list[np.ndarray] = []
    for prompt_index, prompt_builder in enumerate(prompt_builders, start=1):
        prompts = [prompt_builder(noun) for noun in nouns]
        prompt_outputs = []
        total_batches = max(1, (len(prompts) + batch_size - 1) // batch_size)
        for batch_index, start in enumerate(range(0, len(prompts), batch_size), start=1):
            batch_prompts = prompts[start : start + batch_size]
            tokens = bundle.tokenizer(batch_prompts).to(device)
            with torch.no_grad():
                encoded = bundle.model.encode_text(tokens)
                if normalize_prompt_embeddings:
                    encoded = torch.nn.functional.normalize(encoded, dim=-1)
            prompt_outputs.append(encoded.float().cpu().numpy().astype("float32"))
            progress.step(
                f"Text prompt template {prompt_index}/{len(prompt_builders)}",
                batch_index,
                total_batches,
                noun="batch",
            )
        prompt_embeddings.append(np.concatenate(prompt_outputs, axis=0))

    stacked = np.stack(prompt_embeddings, axis=0)
    ensemble_embeddings = np.mean(stacked, axis=0)
    if normalize_ensemble:
        ensemble_embeddings = _normalize_rows(ensemble_embeddings.astype("float32"))
    else:
        ensemble_embeddings = ensemble_embeddings.astype("float32")

    np.savez_compressed(
        cache_path,
        nouns=np.asarray(nouns, dtype=object),
        prompt_embeddings=stacked.astype("float32"),
        ensemble_embeddings=ensemble_embeddings,
    )
    progress.log(f"Saved text prompt bank cache: {cache_path}")
    return nouns, [stacked[index] for index in range(stacked.shape[0])], ensemble_embeddings
