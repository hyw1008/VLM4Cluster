from __future__ import annotations

from vlm4cluster.datasets.base import ImageDatasetSpec, TextDatasetSpec


IMAGE_DATASETS: dict[str, ImageDatasetSpec] = {}
TEXT_DATASETS: dict[str, TextDatasetSpec] = {}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def register_image_dataset(spec: ImageDatasetSpec) -> ImageDatasetSpec:
    IMAGE_DATASETS[spec.name] = spec
    return spec


def register_text_dataset(spec: TextDatasetSpec) -> TextDatasetSpec:
    TEXT_DATASETS[spec.name] = spec
    return spec


def get_image_dataset_spec(name: str) -> ImageDatasetSpec:
    normalized = _normalize_name(name)
    for key, spec in IMAGE_DATASETS.items():
        if _normalize_name(key) == normalized:
            return spec
    try:
        return IMAGE_DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown image dataset '{name}'.") from exc


def get_text_dataset_spec(name: str) -> TextDatasetSpec:
    normalized = _normalize_name(name)
    for key, spec in TEXT_DATASETS.items():
        if _normalize_name(key) == normalized:
            return spec
    try:
        return TEXT_DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown text dataset '{name}'.") from exc


def list_image_dataset_specs() -> list[ImageDatasetSpec]:
    return [IMAGE_DATASETS[key] for key in sorted(IMAGE_DATASETS)]


def list_text_dataset_specs() -> list[TextDatasetSpec]:
    return [TEXT_DATASETS[key] for key in sorted(TEXT_DATASETS)]
