from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from vlm4cluster.config import DatasetConfig
from vlm4cluster.datasets.base import ImageDatasetSpec, LoadedImageDataset
from vlm4cluster.datasets.registry import register_image_dataset
from vlm4cluster.datasets.transforms import build_transform
from vlm4cluster.utils.deps import require_module
from vlm4cluster.utils.progress import get_progress_logger


CIFAR20_CLASS_NAMES = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_sized_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
]

CIFAR100_FINE_TO_CIFAR20 = [
    4,
    1,
    14,
    8,
    0,
    6,
    7,
    7,
    18,
    3,
    3,
    14,
    9,
    18,
    7,
    11,
    3,
    9,
    7,
    11,
    6,
    11,
    5,
    10,
    7,
    6,
    13,
    15,
    3,
    15,
    0,
    11,
    1,
    10,
    12,
    14,
    16,
    9,
    11,
    5,
    5,
    19,
    8,
    8,
    15,
    13,
    14,
    17,
    18,
    10,
    16,
    4,
    17,
    4,
    2,
    0,
    17,
    4,
    18,
    17,
    10,
    3,
    2,
    12,
    12,
    16,
    12,
    1,
    9,
    19,
    2,
    10,
    0,
    1,
    16,
    12,
    9,
    13,
    15,
    13,
    16,
    19,
    2,
    4,
    6,
    19,
    5,
    5,
    8,
    19,
    18,
    1,
    2,
    15,
    6,
    0,
    17,
    8,
    14,
    13,
]


IMAGENET_V2_REVISION = "d626240be2538720e83103a0e1178d24aca8b12c"

IMAGENET_V2_VARIANTS = {
    "matched-frequency": {
        "archive_name": "imagenetv2-matched-frequency.tar.gz",
        "download_url": (
            "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
            f"{IMAGENET_V2_REVISION}/imagenetv2-matched-frequency.tar.gz?download=true"
        ),
    },
    "threshold0.7": {
        "archive_name": "imagenetv2-threshold0.7.tar.gz",
        "download_url": (
            "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
            f"{IMAGENET_V2_REVISION}/imagenetv2-threshold0.7.tar.gz?download=true"
        ),
    },
    "top-images": {
        "archive_name": "imagenetv2-top-images.tar.gz",
        "download_url": (
            "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
            f"{IMAGENET_V2_REVISION}/imagenetv2-top-images.tar.gz?download=true"
        ),
    },
}

IMAGENET_R_ARCHIVE_NAME = "imagenet-r.tar"
IMAGENET_R_DOWNLOAD_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"


def _dataset_root(config: DatasetConfig, dataset_name: str) -> Path:
    return Path(config.root).expanduser() / dataset_name


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _limit_indices(labels: list[int] | None, max_samples: int | None) -> list[int]:
    if labels is None:
        if max_samples is None:
            raise ValueError("max_samples must be set when dataset labels are unavailable.")
        return list(range(max_samples))

    if max_samples is None or max_samples >= len(labels):
        return list(range(len(labels)))

    return list(range(max_samples))


def _torchvision_datasets():
    return require_module("torchvision.datasets", "pip install torch torchvision")


def _torch_data():
    return require_module("torch.utils.data", "pip install torch torchvision")


def _imagefolder_dataset(root: Path, transform):
    datasets = _torchvision_datasets()
    return datasets.ImageFolder(root=str(root), transform=transform)


def _dataset_progress_logger():
    return get_progress_logger("datasets")


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def _iter_class_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")])


def _is_prepared_imagefolder_split(root: Path, *, expected_num_classes: int, expected_prefix: str | None = None) -> bool:
    if not root.exists():
        return False

    class_dirs = _iter_class_dirs(root)
    if len(class_dirs) != expected_num_classes:
        return False

    if expected_prefix is not None and any(not path.name.startswith(expected_prefix) for path in class_dirs):
        return False

    return any(any(child.is_file() for child in class_dir.iterdir()) for class_dir in class_dirs)


def _resolve_imagefolder_class_root(raw_root: Path, *, expected_prefix: str) -> Path | None:
    class_dirs = [path for path in raw_root.iterdir() if path.is_dir() and path.name.startswith(expected_prefix)] if raw_root.exists() else []
    if class_dirs:
        return raw_root

    child_dirs = [path for path in raw_root.iterdir() if path.is_dir()] if raw_root.exists() else []
    if len(child_dirs) == 1:
        child = child_dirs[0]
        child_class_dirs = [path for path in child.iterdir() if path.is_dir() and path.name.startswith(expected_prefix)]
        if child_class_dirs:
            return child

    return None


def _build_manual_imagefolder_dataset(
    config: DatasetConfig,
    *,
    spec_name: str,
    folder_name: str,
    supported_splits: set[str],
) -> LoadedImageDataset:
    split = config.split.lower()
    if split not in supported_splits:
        supported = ", ".join(sorted(supported_splits))
        raise ValueError(f"{spec_name} supports splits: {supported}")

    root = Path(config.root).expanduser() / folder_name / split
    if not root.exists():
        raise FileNotFoundError(
            f"{spec_name} split directory not found: {root}. "
            f"Expected layout: {Path(config.root).expanduser() / folder_name}/<split>/<class_name>/*"
        )

    dataset = _imagefolder_dataset(root, build_transform(config.transform_preset))
    labels = [int(label) for label in dataset.targets]
    return LoadedImageDataset(
        name=spec_name,
        split=split,
        root=root,
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="manual",
    )


def _normalize_imagenet_v2_variant(value: object | None) -> str:
    token = str(value or "matched-frequency").strip().lower()
    token = token.replace("_", "-").replace(" ", "-")
    aliases = {
        "matchedfrequency": "matched-frequency",
        "matched-frequency": "matched-frequency",
        "threshold-0.7": "threshold0.7",
        "threshold0.7": "threshold0.7",
        "threshold0-7": "threshold0.7",
        "topimages": "top-images",
        "top-images": "top-images",
    }
    normalized = aliases.get(token, token)
    if normalized not in IMAGENET_V2_VARIANTS:
        supported = ", ".join(sorted(IMAGENET_V2_VARIANTS))
        raise ValueError(f"Unsupported imagenet_v2_variant '{value}'. Supported variants: {supported}")
    return normalized


def _download_file(url: str, destination: Path) -> Path:
    progress = _dataset_progress_logger()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        progress.log(f"Reusing existing archive: {destination}")
        return destination

    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "VLM4Cluster/imagenet_v2"})
    try:
        with urlopen(request) as response, temp_path.open("wb") as handle:
            total_size_raw = response.headers.get("Content-Length")
            total_size = int(total_size_raw) if total_size_raw and total_size_raw.isdigit() else None
            progress.log(
                f"Downloading archive from {url} to {destination}"
                + (f" ({_format_bytes(total_size)})" if total_size is not None else "")
            )
            chunk_size = 1024 * 1024
            downloaded = 0
            next_fraction_threshold = 0.1
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_size is not None:
                    fraction = downloaded / total_size
                    while fraction >= next_fraction_threshold or downloaded == total_size:
                        percentage = min(100, int(round(next_fraction_threshold * 100)))
                        if downloaded == total_size:
                            percentage = 100
                        progress.log(
                            f"Downloading {destination.name}: {percentage}% "
                            f"({_format_bytes(downloaded)}/{_format_bytes(total_size)})"
                        )
                        if downloaded == total_size:
                            break
                        next_fraction_threshold += 0.1
                elif downloaded == len(chunk) or downloaded % (128 * 1024 * 1024) < len(chunk):
                    progress.log(f"Downloading {destination.name}: downloaded {_format_bytes(downloaded)}")
    except URLError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download ImageNet-V2 archive from {url}: {exc}") from exc

    temp_path.replace(destination)
    progress.log(f"Finished downloading archive: {destination} ({_format_bytes(destination.stat().st_size)})")
    return destination


def _extract_tar_with_progress(archive_path: Path, destination: Path, *, label: str) -> None:
    progress = _dataset_progress_logger()
    progress.log(f"Extracting {label} archive into {destination}")
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        total_members = len(members)
        for index, member in enumerate(members, start=1):
            archive.extract(member, destination)
            progress.step(
                f"Extracting {label}",
                index,
                total_members,
                noun="member",
            )
    progress.log(f"Finished extracting {label} archive into {destination}")


def _resolve_imagenet_v2_archive_path(config: DatasetConfig, *, variant: str) -> Path | None:
    explicit_archive = config.params.get("imagenet_v2_archive_path")
    if explicit_archive is not None:
        candidate = Path(str(explicit_archive)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"imagenet_v2 archive not found: {candidate}")

    archive_name = IMAGENET_V2_VARIANTS[variant]["archive_name"]
    data_root = Path(config.root).expanduser()
    for candidate in (
        data_root / "imagenet_v2" / "_downloads" / archive_name,
        data_root / archive_name,
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_imagenet_r_archive_path(config: DatasetConfig) -> Path | None:
    explicit_archive = config.params.get("imagenet_r_archive_path")
    if explicit_archive is not None:
        candidate = Path(str(explicit_archive)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"imagenet_r archive not found: {candidate}")

    data_root = Path(config.root).expanduser()
    for candidate in (
        data_root / "imagenet_r" / "_downloads" / IMAGENET_R_ARCHIVE_NAME,
        data_root / IMAGENET_R_ARCHIVE_NAME,
        data_root / "imagenet-r.tar",
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_imagenet_v2_numeric_root(raw_root: Path) -> Path | None:
    numeric_dirs = [path for path in raw_root.iterdir() if path.is_dir() and path.name.isdigit()] if raw_root.exists() else []
    if numeric_dirs:
        return raw_root

    child_dirs = [path for path in raw_root.iterdir() if path.is_dir()] if raw_root.exists() else []
    if len(child_dirs) == 1:
        child = child_dirs[0]
        child_numeric_dirs = [path for path in child.iterdir() if path.is_dir() and path.name.isdigit()]
        if child_numeric_dirs:
            return child

    return None


def _resolve_imagenet_reference_root(config: DatasetConfig) -> Path:
    explicit_root = config.params.get("imagenet_v2_reference_root")
    if explicit_root is not None:
        candidate = Path(str(explicit_root)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"imagenet_v2 reference root not found: {candidate}")

    data_root = Path(config.root).expanduser()
    for split in ("val", "train"):
        candidate = data_root / "imagenet" / split
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "imagenet_v2 auto-download needs ImageNet prepared under data/imagenet/val or data/imagenet/train "
        "to map numeric class indices back to wnids."
    )


def _materialize_imagenet_v2(config: DatasetConfig) -> str:
    progress = _dataset_progress_logger()
    data_root = Path(config.root).expanduser()
    target_split_root = data_root / "imagenet_v2" / "test"
    if _is_prepared_imagefolder_split(target_split_root, expected_num_classes=1000, expected_prefix="n"):
        progress.log(f"Reusing prepared ImageNet-V2 split: {target_split_root}")
        return "manual"

    existing_class_dirs = _iter_class_dirs(target_split_root) if target_split_root.exists() else []
    if existing_class_dirs and any(not path.name.startswith("n") for path in existing_class_dirs):
        raise ValueError(
            f"imagenet_v2 target directory already contains non-wnid class folders under {target_split_root}. "
            "Please clean it first or move it aside before auto-materializing ImageNet-V2."
        )

    variant = _normalize_imagenet_v2_variant(config.params.get("imagenet_v2_variant"))
    archive_path = _resolve_imagenet_v2_archive_path(config, variant=variant)
    if archive_path is None:
        if config.download is False:
            raise FileNotFoundError(
                "ImageNet-V2 archive not found locally and image_dataset.download is false. "
                "Either place the archive at data/imagenetv2-<variant>.tar.gz, set "
                "image_dataset.params.imagenet_v2_archive_path, or enable download."
            )
        archive_name = IMAGENET_V2_VARIANTS[variant]["archive_name"]
        archive_path = _download_file(
            str(config.params.get("imagenet_v2_download_url") or IMAGENET_V2_VARIANTS[variant]["download_url"]),
            data_root / "imagenet_v2" / "_downloads" / archive_name,
        )

    raw_root = data_root / "imagenet_v2" / "_raw" / variant
    source_root = _resolve_imagenet_v2_numeric_root(raw_root)
    if source_root is None:
        raw_root.mkdir(parents=True, exist_ok=True)
        _extract_tar_with_progress(archive_path, raw_root, label=f"ImageNet-V2 ({variant})")
        source_root = _resolve_imagenet_v2_numeric_root(raw_root)
    else:
        progress.log(f"Reusing extracted ImageNet-V2 raw directory: {source_root}")
    if source_root is None:
        raise FileNotFoundError(
            f"Could not find numeric ImageNet-V2 class directories after extracting {archive_path} into {raw_root}."
        )

    mode = str(config.params.get("imagenet_v2_materialize_mode", "symlink")).lower()
    if mode not in {"symlink", "copy"}:
        raise ValueError("imagenet_v2_materialize_mode must be either 'symlink' or 'copy'.")

    imagenet_ref = _imagefolder_dataset(_resolve_imagenet_reference_root(config), transform=None)
    idx_to_wnid = list(imagenet_ref.classes)
    if len(idx_to_wnid) != 1000:
        raise ValueError(
            f"Expected 1000 ImageNet reference classes for imagenet_v2 mapping, got {len(idx_to_wnid)}."
        )

    class_dirs = sorted([path for path in source_root.iterdir() if path.is_dir()], key=lambda path: int(path.name))
    target_split_root.mkdir(parents=True, exist_ok=True)
    progress.log(f"Materializing ImageNet-V2 classes into {target_split_root}")
    for index, class_dir in enumerate(class_dirs, start=1):
        if not class_dir.name.isdigit():
            raise ValueError(f"Expected numeric class directory under {source_root}, got '{class_dir.name}'.")

        class_idx = int(class_dir.name)
        if class_idx < 0 or class_idx >= len(idx_to_wnid):
            raise ValueError(f"ImageNet-V2 class index out of range: {class_idx}")

        wnid = idx_to_wnid[class_idx]
        target_class_dir = target_split_root / wnid
        target_class_dir.mkdir(parents=True, exist_ok=True)
        for source_file in class_dir.iterdir():
            if not source_file.is_file():
                continue
            target_file = target_class_dir / source_file.name
            if target_file.exists() or target_file.is_symlink():
                continue
            if mode == "copy":
                shutil.copy2(source_file, target_file)
            else:
                target_file.symlink_to(source_file.resolve())
        progress.step("Materializing ImageNet-V2", index, len(class_dirs), noun="class")

    if not _is_prepared_imagefolder_split(target_split_root, expected_num_classes=1000, expected_prefix="n"):
        raise RuntimeError(
            f"ImageNet-V2 materialization finished but {target_split_root} still does not look complete."
        )

    progress.log(f"Prepared ImageNet-V2 split: {target_split_root}")
    return "python"


def _materialize_imagenet_r(config: DatasetConfig) -> str:
    progress = _dataset_progress_logger()
    data_root = Path(config.root).expanduser()
    target_split_root = data_root / "imagenet_r" / "test"
    if _is_prepared_imagefolder_split(target_split_root, expected_num_classes=200, expected_prefix="n"):
        progress.log(f"Reusing prepared ImageNet-R split: {target_split_root}")
        return "manual"

    existing_class_dirs = _iter_class_dirs(target_split_root) if target_split_root.exists() else []
    if existing_class_dirs and any(not path.name.startswith("n") for path in existing_class_dirs):
        raise ValueError(
            f"imagenet_r target directory already contains non-wnid class folders under {target_split_root}. "
            "Please clean it first or move it aside before auto-materializing ImageNet-R."
        )

    archive_path = _resolve_imagenet_r_archive_path(config)
    if archive_path is None:
        if config.download is False:
            raise FileNotFoundError(
                "ImageNet-R archive not found locally and image_dataset.download is false. "
                "Either place the archive at data/imagenet-r.tar, set "
                "image_dataset.params.imagenet_r_archive_path, or enable download."
            )
        archive_path = _download_file(
            str(config.params.get("imagenet_r_download_url") or IMAGENET_R_DOWNLOAD_URL),
            data_root / "imagenet_r" / "_downloads" / IMAGENET_R_ARCHIVE_NAME,
        )

    raw_root = data_root / "imagenet_r" / "_raw"
    source_root = _resolve_imagefolder_class_root(raw_root, expected_prefix="n")
    if source_root is None:
        raw_root.mkdir(parents=True, exist_ok=True)
        _extract_tar_with_progress(archive_path, raw_root, label="ImageNet-R")
        source_root = _resolve_imagefolder_class_root(raw_root, expected_prefix="n")
    else:
        progress.log(f"Reusing extracted ImageNet-R raw directory: {source_root}")
    if source_root is None:
        raise FileNotFoundError(
            f"Could not find wnid-organized ImageNet-R class directories after extracting {archive_path} into {raw_root}."
        )

    mode = str(config.params.get("imagenet_r_materialize_mode", "symlink")).lower()
    if mode not in {"symlink", "copy"}:
        raise ValueError("imagenet_r_materialize_mode must be either 'symlink' or 'copy'.")

    target_split_root.mkdir(parents=True, exist_ok=True)
    source_class_dirs = _iter_class_dirs(source_root)
    progress.log(f"Materializing ImageNet-R classes into {target_split_root}")
    for index, source_class_dir in enumerate(source_class_dirs, start=1):
        target_class_dir = target_split_root / source_class_dir.name
        if target_class_dir.exists() or target_class_dir.is_symlink():
            progress.step("Materializing ImageNet-R", index, len(source_class_dirs), noun="class")
            continue
        if mode == "copy":
            shutil.copytree(source_class_dir, target_class_dir)
        else:
            target_class_dir.symlink_to(source_class_dir.resolve(), target_is_directory=True)
        progress.step("Materializing ImageNet-R", index, len(source_class_dirs), noun="class")

    if not _is_prepared_imagefolder_split(target_split_root, expected_num_classes=200, expected_prefix="n"):
        raise RuntimeError(
            f"ImageNet-R materialization finished but {target_split_root} still does not look complete."
        )

    progress.log(f"Prepared ImageNet-R split: {target_split_root}")
    return "python"


def _resolve_imagenet_subset_wnid_path(config: DatasetConfig, subset_name: str) -> Path:
    param_key = f"{subset_name}_wnids_path"
    explicit_path = config.params.get(param_key) or config.params.get("imagenet_subset_wnids_path")
    if explicit_path is not None:
        candidate = Path(str(explicit_path)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"{subset_name} wnid list not found: {candidate}")

    root_candidate = Path(config.root).expanduser() / "imagenet_subsets" / f"{subset_name}_wnids.txt"
    if root_candidate.exists():
        return root_candidate.resolve()

    package_candidate = _project_root() / "data" / "imagenet_subsets" / f"{subset_name}_wnids.txt"
    if package_candidate.exists():
        return package_candidate.resolve()

    raise FileNotFoundError(
        f"{subset_name} wnid list not found. Expected {root_candidate} or {package_candidate}, "
        f"or set image_dataset.params.{param_key}."
    )


def _read_wnids(path: Path) -> list[str]:
    wnids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            wnid = line.strip()
            if wnid and not wnid.startswith("#"):
                wnids.append(wnid)
    if not wnids:
        raise ValueError(f"No wnids found in {path}")
    return wnids


def _materialize_imagenet_subset(config: DatasetConfig, *, subset_name: str, split: str) -> None:
    data_root = Path(config.root).expanduser()
    source_split_root = data_root / "imagenet" / split
    target_split_root = data_root / subset_name / split
    wnid_path = _resolve_imagenet_subset_wnid_path(config, subset_name)
    wnids = _read_wnids(wnid_path)
    if target_split_root.exists() and all((target_split_root / wnid).exists() for wnid in wnids):
        return

    mode = str(config.params.get("imagenet_subset_mode", "symlink")).lower()
    if mode not in {"symlink", "copy"}:
        raise ValueError("imagenet_subset_mode must be either 'symlink' or 'copy'.")

    if not source_split_root.exists():
        raise FileNotFoundError(
            f"Cannot auto-generate {subset_name}: source ImageNet split directory not found: {source_split_root}. "
            "Please prepare data/imagenet/<split>/<wnid> first."
        )

    missing_sources = [wnid for wnid in wnids if not (source_split_root / wnid).exists()]
    if missing_sources:
        preview = ", ".join(missing_sources[:10])
        raise FileNotFoundError(
            f"Cannot auto-generate {subset_name}/{split}: {len(missing_sources)} source wnid directories are missing "
            f"under {source_split_root}. Missing examples: {preview}"
        )

    target_split_root.mkdir(parents=True, exist_ok=True)
    for wnid in wnids:
        source_dir = source_split_root / wnid
        target_dir = target_split_root / wnid
        if target_dir.exists() or target_dir.is_symlink():
            continue
        if mode == "copy":
            shutil.copytree(source_dir, target_dir)
        else:
            target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)


def _build_imagenet_subset_dataset(
    config: DatasetConfig,
    *,
    subset_name: str,
    supported_splits: set[str],
) -> LoadedImageDataset:
    split = config.split.lower()
    if split not in supported_splits:
        supported = ", ".join(sorted(supported_splits))
        raise ValueError(f"{subset_name} supports splits: {supported}")

    _materialize_imagenet_subset(config, subset_name=subset_name, split=split)

    return _build_manual_imagefolder_dataset(
        config,
        spec_name=subset_name,
        folder_name=subset_name,
        supported_splits=supported_splits,
    )


def _build_cifar10(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    if split not in {"train", "test"}:
        raise ValueError("cifar10 supports splits: train, test")
    dataset = datasets.CIFAR10(
        root=str(_dataset_root(config, "cifar10")),
        train=split == "train",
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset.targets]
    return LoadedImageDataset(
        name="cifar10",
        split=split,
        root=_dataset_root(config, "cifar10"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
    )


def _build_cifar100(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    if split not in {"train", "test"}:
        raise ValueError("cifar100 supports splits: train, test")
    dataset = datasets.CIFAR100(
        root=str(_dataset_root(config, "cifar100")),
        train=split == "train",
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset.targets]
    return LoadedImageDataset(
        name="cifar100",
        split=split,
        root=_dataset_root(config, "cifar100"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
    )


def _build_cifar20(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    if split not in {"train", "test"}:
        raise ValueError("cifar20 supports splits: train, test")
    dataset = datasets.CIFAR100(
        root=str(_dataset_root(config, "cifar20")),
        train=split == "train",
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [CIFAR100_FINE_TO_CIFAR20[int(label)] for label in dataset.targets]
    return LoadedImageDataset(
        name="cifar20",
        split=split,
        root=_dataset_root(config, "cifar20"),
        dataset=dataset,
        class_names=CIFAR20_CLASS_NAMES,
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
        metadata={"source_dataset": "cifar100"},
    )


def _build_stl10(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    supported = {"train", "test", "train+unlabeled"}
    if split not in supported:
        raise ValueError("stl10 supports splits: train, test, train+unlabeled")
    dataset = datasets.STL10(
        root=str(_dataset_root(config, "stl10")),
        split=split,
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = dataset.labels.tolist() if getattr(dataset, "labels", None) is not None else None
    return LoadedImageDataset(
        name="stl10",
        split=split,
        root=_dataset_root(config, "stl10"),
        dataset=dataset,
        class_names=list(getattr(dataset, "classes", [])),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
    )


def _build_imagenet(config: DatasetConfig) -> LoadedImageDataset:
    if config.download:
        raise ValueError("imagenet must be manually prepared under data/imagenet.")

    datasets = _torchvision_datasets()
    split = config.split.lower()
    if split not in {"train", "val"}:
        raise ValueError("imagenet supports splits: train, val")

    root = _dataset_root(config, "imagenet") / split
    if not root.exists():
        raise FileNotFoundError(
            f"ImageNet split directory not found: {root}. "
            "Expected layout: data/imagenet/train/<class_name>/* and data/imagenet/val/<class_name>/*"
        )

    dataset = datasets.ImageFolder(root=str(root), transform=build_transform(config.transform_preset))
    labels = [int(label) for label in dataset.targets]
    return LoadedImageDataset(
        name="imagenet",
        split=split,
        root=root,
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="manual",
    )


def _build_imagenet10(config: DatasetConfig) -> LoadedImageDataset:
    return _build_imagenet_subset_dataset(
        config,
        subset_name="imagenet10",
        supported_splits={"train", "val"},
    )


def _build_imagenet_dogs(config: DatasetConfig) -> LoadedImageDataset:
    return _build_imagenet_subset_dataset(
        config,
        subset_name="imagenet_dogs",
        supported_splits={"train", "val"},
    )


def _build_imagenet_a(config: DatasetConfig) -> LoadedImageDataset:
    return _build_manual_imagefolder_dataset(
        config,
        spec_name="imagenet_a",
        folder_name="imagenet_a",
        supported_splits={"test"},
    )


def _build_imagenet_sketch(config: DatasetConfig) -> LoadedImageDataset:
    return _build_manual_imagefolder_dataset(
        config,
        spec_name="imagenet_sketch",
        folder_name="imagenet_sketch",
        supported_splits={"test"},
    )


def _build_imagenet_r(config: DatasetConfig) -> LoadedImageDataset:
    materialize_mode = _materialize_imagenet_r(config)
    dataset = _build_manual_imagefolder_dataset(
        config,
        spec_name="imagenet_r",
        folder_name="imagenet_r",
        supported_splits={"test"},
    )
    dataset.download_mode = materialize_mode
    return dataset


def _build_imagenet_v2(config: DatasetConfig) -> LoadedImageDataset:
    materialize_mode = _materialize_imagenet_v2(config)
    dataset = _build_manual_imagefolder_dataset(
        config,
        spec_name="imagenet_v2",
        folder_name="imagenet_v2",
        supported_splits={"test"},
    )
    dataset.download_mode = materialize_mode
    dataset.metadata["variant"] = _normalize_imagenet_v2_variant(config.params.get("imagenet_v2_variant"))
    return dataset


def _build_imagenet_c(config: DatasetConfig) -> LoadedImageDataset:
    return _build_manual_imagefolder_dataset(
        config,
        spec_name="imagenet_c",
        folder_name="imagenet_c/gaussian_noise/1",
        supported_splits={"test"},
    )


def _build_ucf101(config: DatasetConfig) -> LoadedImageDataset:
    return _build_manual_imagefolder_dataset(
        config,
        spec_name="ucf101",
        folder_name="ucf101",
        supported_splits={"train", "val", "test"},
    )


def _build_dtd(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    supported_splits = {"train", "val", "test", "trainval"}
    if split not in supported_splits:
        supported = ", ".join(sorted(supported_splits))
        raise ValueError(f"dtd supports splits: {supported}")

    root = _dataset_root(config, "dtd")
    transform = build_transform(config.transform_preset)
    download = config.download if config.download is not None else True

    if split == "trainval":
        data_mod = _torch_data()
        train_dataset = datasets.DTD(root=str(root), split="train", download=download, transform=transform)
        val_dataset = datasets.DTD(root=str(root), split="val", download=download, transform=transform)
        dataset = data_mod.ConcatDataset([train_dataset, val_dataset])
        labels = [int(label) for label in train_dataset._labels] + [int(label) for label in val_dataset._labels]
        class_names = list(train_dataset.classes)
    else:
        dataset = datasets.DTD(root=str(root), split=split, download=download, transform=transform)
        labels = [int(label) for label in dataset._labels]
        class_names = list(dataset.classes)

    return LoadedImageDataset(
        name="dtd",
        split=split,
        root=root,
        dataset=dataset,
        class_names=class_names,
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python" if split != "trainval" else "python",
    )


def _build_places365_standard(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    split_mapping = {"train": "train-standard", "train-standard": "train-standard", "val": "val"}
    if split not in split_mapping:
        raise ValueError("places365_standard supports splits: train, train-standard, val")

    dataset = datasets.Places365(
        root=str(_dataset_root(config, "places365_standard")),
        split=split_mapping[split],
        small=False,
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset.targets]
    return LoadedImageDataset(
        name="places365_standard",
        split=split,
        root=_dataset_root(config, "places365_standard"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
        metadata={"torchvision_split": split_mapping[split]},
    )


def _build_aircraft(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    supported_splits = {"train", "val", "trainval", "test"}
    if split not in supported_splits:
        supported = ", ".join(sorted(supported_splits))
        raise ValueError(f"aircraft supports splits: {supported}")

    dataset = datasets.FGVCAircraft(
        root=str(_dataset_root(config, "aircraft")),
        split=split,
        annotation_level="variant",
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset._labels]
    return LoadedImageDataset(
        name="aircraft",
        split=split,
        root=_dataset_root(config, "aircraft"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
    )


def _build_flowers(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    supported_splits = {"train", "val", "test"}
    if split not in supported_splits:
        supported = ", ".join(sorted(supported_splits))
        raise ValueError(f"flowers supports splits: {supported}")

    dataset = datasets.Flowers102(
        root=str(_dataset_root(config, "flowers")),
        split=split,
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset._labels]
    class_names = [f"class_{index}" for index in range(102)]
    return LoadedImageDataset(
        name="flowers",
        split=split,
        root=_dataset_root(config, "flowers"),
        dataset=dataset,
        class_names=class_names,
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
    )


def _build_food(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    if split not in {"train", "test"}:
        raise ValueError("food supports splits: train, test")

    dataset = datasets.Food101(
        root=str(_dataset_root(config, "food")),
        split=split,
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset._labels]
    return LoadedImageDataset(
        name="food",
        split=split,
        root=_dataset_root(config, "food"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
    )


def _build_pets(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    split_mapping = {"train": "trainval", "trainval": "trainval", "test": "test"}
    if split not in split_mapping:
        raise ValueError("pets supports splits: train, trainval, test")

    dataset = datasets.OxfordIIITPet(
        root=str(_dataset_root(config, "pets")),
        split=split_mapping[split],
        target_types="category",
        download=config.download if config.download is not None else True,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(label) for label in dataset._labels]
    return LoadedImageDataset(
        name="pets",
        split=split,
        root=_dataset_root(config, "pets"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="python",
        metadata={"torchvision_split": split_mapping[split]},
    )


def _build_cars(config: DatasetConfig) -> LoadedImageDataset:
    datasets = _torchvision_datasets()
    split = config.split.lower()
    if split not in {"train", "test"}:
        raise ValueError("cars supports splits: train, test")
    if config.download:
        raise ValueError("cars must be manually prepared; automatic download is unavailable.")

    imagefolder_root = _dataset_root(config, "cars") / split
    if imagefolder_root.exists():
        return _build_manual_imagefolder_dataset(
            config,
            spec_name="cars",
            folder_name="cars",
            supported_splits={"train", "test"},
        )

    dataset = datasets.StanfordCars(
        root=str(_dataset_root(config, "cars")),
        split=split,
        download=False,
        transform=build_transform(config.transform_preset),
    )
    labels = [int(target) for _, target in dataset._samples]
    return LoadedImageDataset(
        name="cars",
        split=split,
        root=_dataset_root(config, "cars"),
        dataset=dataset,
        class_names=list(dataset.classes),
        labels=labels,
        sample_indices=_limit_indices(labels, config.max_samples),
        download_mode="manual",
    )


register_image_dataset(
    ImageDatasetSpec(
        name="cifar10",
        description="Torchvision CIFAR-10 dataset.",
        download_mode="python",
        supported_splits=("train", "test"),
        builder=_build_cifar10,
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="cifar20",
        description="CIFAR-20 via CIFAR-100 coarse-label remapping.",
        download_mode="python",
        supported_splits=("train", "test"),
        builder=_build_cifar20,
        notes="Internally downloads CIFAR-100 and maps fine labels to 20 superclasses.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="cifar100",
        description="Torchvision CIFAR-100 dataset.",
        download_mode="python",
        supported_splits=("train", "test"),
        builder=_build_cifar100,
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="stl10",
        description="Torchvision STL-10 dataset.",
        download_mode="python",
        supported_splits=("train", "test", "train+unlabeled"),
        builder=_build_stl10,
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet",
        description="Manual ImageNet folder using ImageFolder layout.",
        download_mode="manual",
        supported_splits=("train", "val"),
        builder=_build_imagenet,
        notes="Place files under data/imagenet/<split>/<class_name>/* before running.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet10",
        description="Manual ImageNet-10 subset using ImageFolder layout.",
        download_mode="manual",
        supported_splits=("train", "val"),
        builder=_build_imagenet10,
        notes=(
            "Auto-generated from data/imagenet/<split>/<wnid> using "
            "data/imagenet_subsets/imagenet10_wnids.txt when missing."
        ),
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet_dogs",
        description="Manual ImageNet-Dogs subset using ImageFolder layout.",
        download_mode="manual",
        supported_splits=("train", "val"),
        builder=_build_imagenet_dogs,
        notes=(
            "Auto-generated from data/imagenet/<split>/<wnid> using "
            "data/imagenet_subsets/imagenet_dogs_wnids.txt when missing."
        ),
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet_a",
        description="Prepared ImageNet-A target set using ImageFolder layout.",
        download_mode="external_scripted",
        supported_splits=("test",),
        builder=_build_imagenet_a,
        notes="Expected under data/imagenet_a/test/<class_name>/* for domain-shift evaluation.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet_sketch",
        description="Prepared ImageNet-Sketch target set using ImageFolder layout.",
        download_mode="external_scripted",
        supported_splits=("test",),
        builder=_build_imagenet_sketch,
        notes="Expected under data/imagenet_sketch/test/<class_name>/* for domain-shift evaluation.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet_r",
        description="ImageNet-R target set using ImageFolder layout, with optional in-code auto-download.",
        download_mode="python_optional",
        supported_splits=("test",),
        builder=_build_imagenet_r,
        notes=(
            "Expected under data/imagenet_r/test/<class_name>/* for domain-shift evaluation. "
            "If missing, the loader can auto-download the official archive and materialize wnid folders "
            "under data/imagenet_r/test."
        ),
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet_v2",
        description="ImageNet-V2 target set using ImageFolder layout, with optional in-code auto-download.",
        download_mode="python_optional",
        supported_splits=("test",),
        builder=_build_imagenet_v2,
        notes=(
            "Expected under data/imagenet_v2/test/<class_name>/* for domain-shift evaluation. "
            "If missing, the loader can auto-download matched-frequency by default and map numeric "
            "class indices back to wnids using data/imagenet/{val|train}."
        ),
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="imagenet_c",
        description="Prepared ImageNet-C gaussian_noise severity-1 target set using ImageFolder layout.",
        download_mode="manual",
        supported_splits=("test",),
        builder=_build_imagenet_c,
        notes="Expected under data/imagenet_c/gaussian_noise/1/test/<class_name>/* for domain-shift evaluation.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="dtd",
        description="Torchvision DTD dataset.",
        download_mode="python",
        supported_splits=("train", "val", "test", "trainval"),
        builder=_build_dtd,
        notes="For TAC, the paper trains on train+val and evaluates on test.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="places365_standard",
        description="Torchvision Places365 Standard dataset.",
        download_mode="python",
        supported_splits=("train", "train-standard", "val"),
        builder=_build_places365_standard,
        notes="Maps benchmark split 'train' to torchvision's 'train-standard'.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="aircraft",
        description="Torchvision FGVC Aircraft dataset at variant level.",
        download_mode="python",
        supported_splits=("train", "val", "trainval", "test"),
        builder=_build_aircraft,
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="flowers",
        description="Torchvision Flowers102 dataset.",
        download_mode="python",
        supported_splits=("train", "val", "test"),
        builder=_build_flowers,
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="food",
        description="Torchvision Food101 dataset.",
        download_mode="python",
        supported_splits=("train", "test"),
        builder=_build_food,
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="pets",
        description="Torchvision Oxford-IIIT Pet dataset.",
        download_mode="python",
        supported_splits=("train", "trainval", "test"),
        builder=_build_pets,
        notes="Maps benchmark split 'train' to torchvision's 'trainval'.",
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="cars",
        description="Manually prepared Stanford Cars dataset from ImageFolder train/test folders or torchvision metadata files.",
        download_mode="manual",
        supported_splits=("train", "test"),
        builder=_build_cars,
        notes=(
            "Supports official-style data/cars/<split>/<class_name>/* folders and falls back to "
            "torchvision's stanford_cars metadata layout under data/cars/stanford_cars."
        ),
    )
)

register_image_dataset(
    ImageDatasetSpec(
        name="ucf101",
        description="Prepared UCF101 frame dataset using ImageFolder layout.",
        download_mode="manual",
        supported_splits=("train", "val", "test"),
        builder=_build_ucf101,
        notes="Expected as image folders under data/ucf101/<split>/<class_name>/* after frame preparation.",
    )
)
