from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import gc
import hashlib
import json
import shutil

import numpy as np

from vlm4cluster.config import DatasetAttackConfig, DatasetConfig
from vlm4cluster.datasets.base import LoadedImageDataset, get_image_label
from vlm4cluster.datasets.transforms import build_transform
from vlm4cluster.methods.feature_cache import cache_key, cache_token
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.efficiency import suspend_efficiency_memory
from vlm4cluster.utils.io import ensure_dir
from vlm4cluster.utils.progress import get_progress_logger


HF_CHECKPOINT_URL = "https://huggingface.co/jiamingzz/anyattack/tree/main/checkpoints"
DEFAULT_VARIANT = "anyattack_cos"


class EfficientAttention(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_channels: int, key_channels: int, head_count: int, value_channels: int) -> None:
        torch_nn = require_module("torch.nn", "pip install torch")
        torch_functional = require_module("torch.nn.functional", "pip install torch")
        super().__init__()
        self.functional = torch_functional
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.keys = torch_nn.Conv2d(in_channels, key_channels, 1)
        self.queries = torch_nn.Conv2d(in_channels, key_channels, 1)
        self.values = torch_nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = torch_nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []
        for index in range(self.head_count):
            key = self.functional.softmax(
                keys[:, index * head_key_channels : (index + 1) * head_key_channels, :],
                dim=2,
            )
            query = self.functional.softmax(
                queries[:, index * head_key_channels : (index + 1) * head_key_channels, :],
                dim=1,
            )
            value = values[:, index * head_value_channels : (index + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)
        torch = require_module("torch", "pip install torch")
        aggregated_values = torch.cat(attended_values, dim=1)
        return self.reprojection(aggregated_values) + input_


class ResBlock(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_channels: int, out_channels: int, key_channels: int, head_count: int, value_channels: int):
        torch_nn = require_module("torch.nn", "pip install torch")
        super().__init__()
        self.conv1 = torch_nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch_nn.BatchNorm2d(out_channels)
        self.conv2 = torch_nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch_nn.BatchNorm2d(out_channels)
        self.activation = torch_nn.LeakyReLU(0.2, inplace=True)
        self.attention = EfficientAttention(out_channels, key_channels, head_count, value_channels)
        self.skip_conv = torch_nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else torch_nn.Identity()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.skip_conv(residual)
        return self.activation(out)


class UpBlock(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_channels: int, out_channels: int):
        torch_nn = require_module("torch.nn", "pip install torch")
        super().__init__()
        self.up = torch_nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = torch_nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = torch_nn.BatchNorm2d(out_channels)
        self.activation = torch_nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(self.up(x))))


class AnyAttackDecoder(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, embed_dim: int = 512, img_channels: int = 3, img_size: int = 224) -> None:
        torch_nn = require_module("torch.nn", "pip install torch")
        super().__init__()
        self.init_size = img_size // 16
        self.fc = torch_nn.Sequential(torch_nn.Linear(embed_dim, 256 * self.init_size**2))
        self.upsample_blocks = torch_nn.ModuleList(
            [
                ResBlock(256, 256, 64, 8, 256),
                UpBlock(256, 128),
                ResBlock(128, 128, 32, 8, 128),
                UpBlock(128, 64),
                ResBlock(64, 64, 16, 8, 64),
                UpBlock(64, 32),
                ResBlock(32, 32, 8, 8, 32),
                UpBlock(32, 16),
                ResBlock(16, 16, 4, 8, 16),
            ]
        )
        self.final_conv = torch_nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), -1)
        out = self.fc(embedding.float())
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        for block in self.upsample_blocks:
            out = block(out)
        return self.final_conv(out)


class MaterializedAdversarialDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(self, samples: list[dict[str, Any]], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_mod = require_module("PIL.Image", "pip install pillow")
        sample = self.samples[index]
        image = image_mod.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample["label"])


class AnyAttackPairDataset(require_module("torch.utils.data", "pip install torch torchvision").Dataset):
    def __init__(
        self,
        dataset: LoadedImageDataset,
        labels: list[int],
        target_positions: list[int],
    ) -> None:
        self.dataset = dataset
        self.labels = labels
        self.target_positions = target_positions

    def __len__(self) -> int:
        return len(self.dataset.sample_indices)

    def __getitem__(self, position: int):
        source_index = self.dataset.sample_indices[position]
        target_index = self.dataset.sample_indices[self.target_positions[position]]
        source_image, _ = self.dataset.dataset[source_index]
        target_image, _ = self.dataset.dataset[target_index]
        return {
            "position": int(position),
            "source_index": int(source_index),
            "target_index": int(target_index),
            "label": int(self.labels[position]),
            "source_tensor": _load_image_as_tensor(source_image),
            "target_tensor": _load_image_as_tensor(target_image),
        }


def _sha1_text(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _sample_view_token(sample_indices: list[int]) -> str:
    digest = _sha1_text(",".join(str(index) for index in sample_indices))
    return f"samples{len(sample_indices)}-{digest}"


def anyattack_cache_token(attack: DatasetAttackConfig, *, view_token: str | None = None) -> str:
    checkpoint = Path(attack.decoder_path).stem
    eps_token = f"eps{int(round(float(attack.eps) * 255))}-255"
    token = cache_key(
        "attack-anyattack",
        f"variant-{attack.variant or DEFAULT_VARIANT}",
        f"ckpt-{checkpoint}",
        eps_token,
        f"target-{attack.target_strategy.strip().lower()}",
        f"seed{attack.seed}",
    )
    return cache_key(token, view_token) if view_token is not None else token


def _load_image_as_tensor(image):
    transforms = require_module("torchvision.transforms", "pip install torch torchvision")
    if hasattr(image, "detach"):
        image = transforms.ToPILImage()(image)
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )(image.convert("RGB") if hasattr(image, "convert") else image)


def _torch_load(path: Path, map_location: str):
    torch = require_module("torch", "pip install torch")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_decoder(decoder_path: Path, device):
    torch = require_module("torch", "pip install torch")
    decoder = AnyAttackDecoder(embed_dim=512).to(device).eval()
    payload = _torch_load(decoder_path, map_location="cpu")
    state_dict = payload.get("decoder_state_dict") if isinstance(payload, dict) else None
    if state_dict is None:
        raise ValueError(f"AnyAttack checkpoint does not contain 'decoder_state_dict': {decoder_path}")
    normalized_state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    decoder.load_state_dict(normalized_state_dict)
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    return decoder


def _load_clip_image_encoder(device):
    open_clip = require_module("open_clip", "pip install open_clip_torch")
    torch = require_module("torch", "pip install torch")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return model, mean, std


def _select_target_positions(labels: list[int] | None, *, seed: int, avoid_same_label: bool) -> list[int]:
    sample_count = len(labels) if labels is not None else 0
    if sample_count == 0:
        return []
    rng = np.random.default_rng(seed)
    if labels is None or not avoid_same_label:
        targets = rng.permutation(sample_count)
        if sample_count > 1:
            same = np.flatnonzero(targets == np.arange(sample_count))
            if same.size:
                targets[same] = np.roll(targets[same], 1)
        return targets.astype(np.int64).tolist()

    label_to_positions: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        label_to_positions.setdefault(int(label), []).append(position)
    unique_labels = sorted(label_to_positions)
    if len(unique_labels) <= 1:
        return _select_target_positions(labels, seed=seed, avoid_same_label=False)

    targets = []
    for label in labels:
        candidate_labels = [candidate for candidate in unique_labels if candidate != int(label)]
        target_label = int(rng.choice(candidate_labels))
        targets.append(int(rng.choice(label_to_positions[target_label])))
    return targets


def _manifest_matches(manifest_path: Path, *, expected: dict[str, Any], expected_count: int) -> bool:
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("fingerprint") != expected.get("fingerprint"):
        return False
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_count:
        return False
    return all(Path(str(sample.get("path", ""))).exists() for sample in samples)


def _read_manifest_samples(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return [
        {
            **sample,
            "path": str(Path(sample["path"]).expanduser()),
        }
        for sample in manifest["samples"]
    ]


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(manifest_path)


def _attack_metadata(
    attack: DatasetAttackConfig,
    *,
    materialized_root: Path,
    view_token: str,
    target_strategy: str,
) -> dict[str, Any]:
    token = anyattack_cache_token(attack, view_token=view_token)
    return {
        "name": "anyattack",
        "variant": attack.variant or DEFAULT_VARIANT,
        "cache_token": token,
        "decoder_path": str(Path(attack.decoder_path).expanduser()),
        "decoder_checkpoint": Path(attack.decoder_path).stem,
        "eps": float(attack.eps),
        "target_strategy": target_strategy,
        "avoid_same_label": bool(attack.avoid_same_label),
        "seed": int(attack.seed),
        "materialized_root": str(materialized_root),
        "source": "AnyAttack-Cos" if (attack.variant or DEFAULT_VARIANT) == DEFAULT_VARIANT else attack.variant,
    }


def _generate_anyattack_images(
    *,
    dataset: LoadedImageDataset,
    dataset_config: DatasetConfig,
    attack: DatasetAttackConfig,
    labels: list[int],
    target_positions: list[int],
    decoder_path: Path,
    materialized_root: Path,
    manifest_path: Path,
    fingerprint: str,
    view_token: str,
    target_strategy: str,
    torch,
    torchvision_utils,
    data_mod,
    progress,
) -> list[dict[str, Any]]:
    if materialized_root.exists():
        shutil.rmtree(materialized_root)
    ensure_dir(materialized_root)
    progress.log(
        f"Materializing AnyAttack-Cos images for {dataset.name}/{dataset.split} "
        f"into {materialized_root}"
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = str(
        attack.params.get(
            "device",
            dataset_config.params.get("attack_device", default_device),
        )
    )
    device = torch.device(device_name)
    clip_model, clip_mean, clip_std = _load_clip_image_encoder(device)
    decoder = _load_decoder(decoder_path, device)

    samples: list[dict[str, Any]] = []
    batch_size = max(1, int(attack.batch_size))
    pair_dataset = AnyAttackPairDataset(dataset, labels, target_positions)
    pair_loader = data_mod.DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(attack.params.get("num_workers", 0)),
    )
    total_batches = max(1, len(pair_loader))
    with torch.no_grad():
        for batch_index, batch in enumerate(pair_loader, start=1):
            clean_images = batch["source_tensor"].to(device)
            target_images = batch["target_tensor"].to(device)
            target_embeddings = clip_model.encode_image((target_images - clip_mean) / clip_std)
            noise = torch.clamp(decoder(target_embeddings), -float(attack.eps), float(attack.eps))
            adv_images = torch.clamp(clean_images + noise, 0.0, 1.0)
            batch_size_actual = int(adv_images.shape[0])
            for offset in range(batch_size_actual):
                row = {
                    "position": int(batch["position"][offset]),
                    "source_index": int(batch["source_index"][offset]),
                    "target_index": int(batch["target_index"][offset]),
                    "label": int(batch["label"][offset]),
                }
                if 0 <= row["label"] < len(dataset.class_names):
                    class_name = dataset.class_names[row["label"]]
                else:
                    class_name = f"class_{row['label']}"
                safe_class_name = f"{int(row['label']):05d}_{cache_token(class_name)}"
                class_dir = ensure_dir(materialized_root / safe_class_name)
                image_path = (
                    class_dir
                    / f"{row['position']:08d}__src{row['source_index']}__tgt{row['target_index']}.png"
                )
                torchvision_utils.save_image(adv_images[offset].cpu(), image_path)
                samples.append(
                    {
                        "path": str(image_path),
                        "label": int(row["label"]),
                        "source_index": int(row["source_index"]),
                        "target_index": int(row["target_index"]),
                        "class_name": class_name,
                    }
                )
            progress.step("AnyAttack image generation", batch_index, total_batches, noun="batch")

    manifest = {
        "fingerprint": fingerprint,
        "attack": _attack_metadata(
            attack,
            materialized_root=materialized_root,
            view_token=view_token,
            target_strategy=target_strategy,
        ),
        "source_dataset": {
            "name": dataset.name,
            "split": dataset.split,
            "root": str(dataset.root),
            "sample_count": len(dataset.sample_indices),
        },
        "samples": samples,
    }
    _write_manifest(manifest_path, manifest)
    progress.log(f"Saved AnyAttack manifest: {manifest_path}")
    return samples


def materialize_anyattack_dataset(
    dataset: LoadedImageDataset,
    dataset_config: DatasetConfig,
    attack: DatasetAttackConfig,
) -> LoadedImageDataset:
    torch = require_module("torch", "pip install torch")
    torchvision_utils = require_module("torchvision.utils", "pip install torch torchvision")
    data_mod = require_module("torch.utils.data", "pip install torch torchvision")
    progress = get_progress_logger("anyattack", asdict(attack))

    decoder_path = Path(attack.decoder_path).expanduser()
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"AnyAttack decoder checkpoint not found: {decoder_path}. "
            f"Download the AnyAttack-Cos checkpoint (coco_cos.pt) from {HF_CHECKPOINT_URL} "
            "or set image_dataset.attack.decoder_path."
        )
    target_strategy = attack.target_strategy.strip().lower()
    if target_strategy != "shuffle":
        raise ValueError("AnyAttack currently supports target_strategy='shuffle'.")

    if dataset.labels is not None:
        labels = [int(dataset.labels[index]) for index in dataset.sample_indices]
    else:
        labels = [get_image_label(dataset, index, dataset.dataset[index][1]) for index in dataset.sample_indices]
    target_positions = _select_target_positions(
        labels,
        seed=int(attack.seed),
        avoid_same_label=bool(attack.avoid_same_label),
    )
    view_token = _sample_view_token(dataset.sample_indices)
    attack_token = anyattack_cache_token(attack, view_token=view_token)
    output_root = Path(str(attack.params.get("output_root", Path(dataset_config.root).expanduser() / "adversarial"))).expanduser()
    materialized_root = output_root / "anyattack" / dataset.name / dataset.split / attack_token
    manifest_path = materialized_root / "manifest.json"

    fingerprint = _sha1_text(
        json.dumps(
            {
                "dataset": dataset.name,
                "split": dataset.split,
                "source_root": str(dataset.root),
                "sample_indices": dataset.sample_indices,
                "labels": labels,
                "targets": target_positions,
                "attack": {
                    "variant": attack.variant,
                    "decoder_path": str(decoder_path.resolve()),
                    "eps": float(attack.eps),
                    "target_strategy": target_strategy,
                    "avoid_same_label": bool(attack.avoid_same_label),
                    "seed": int(attack.seed),
                },
            },
            sort_keys=True,
        )
    )
    expected_manifest = {"fingerprint": fingerprint}
    if _manifest_matches(manifest_path, expected=expected_manifest, expected_count=len(dataset.sample_indices)):
        progress.log(f"Reusing materialized AnyAttack images: {materialized_root}")
        samples = _read_manifest_samples(manifest_path)
    else:
        with suspend_efficiency_memory():
            samples = _generate_anyattack_images(
                dataset=dataset,
                dataset_config=dataset_config,
                attack=attack,
                labels=labels,
                target_positions=target_positions,
                decoder_path=decoder_path,
                materialized_root=materialized_root,
                manifest_path=manifest_path,
                fingerprint=fingerprint,
                view_token=view_token,
                target_strategy=target_strategy,
                torch=torch,
                torchvision_utils=torchvision_utils,
                data_mod=data_mod,
                progress=progress,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metadata = dict(dataset.metadata)
    metadata["attack"] = _attack_metadata(
        attack,
        materialized_root=materialized_root,
        view_token=view_token,
        target_strategy=target_strategy,
    )
    metadata["source_dataset_root"] = str(dataset.root)
    metadata["benchmark_data_root"] = str(Path(dataset_config.root).expanduser())
    metadata["manifest_path"] = str(manifest_path)

    materialized_dataset = MaterializedAdversarialDataset(
        samples,
        transform=build_transform(dataset_config.transform_preset),
    )
    return LoadedImageDataset(
        name=dataset.name,
        split=dataset.split,
        root=materialized_root,
        dataset=materialized_dataset,
        class_names=list(dataset.class_names),
        labels=[int(sample["label"]) for sample in samples],
        sample_indices=list(range(len(samples))),
        download_mode=dataset.download_mode,
        metadata=metadata,
    )
