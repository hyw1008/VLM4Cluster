from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vlm4cluster.config import DatasetConfig


ImageDatasetBuilder = Callable[[DatasetConfig], "LoadedImageDataset"]
TextDatasetBuilder = Callable[[DatasetConfig, list[str]], "LoadedTextDataset"]


@dataclass(slots=True)
class LoadedImageDataset:
    name: str
    split: str
    root: Path
    dataset: Any
    class_names: list[str]
    labels: list[int] | None
    sample_indices: list[int]
    download_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.sample_indices)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def get_image_label(dataset: LoadedImageDataset, source_index: int, raw_label: Any) -> int:
    if dataset.labels is not None:
        return int(dataset.labels[source_index])
    return int(raw_label)


def get_image_labels_for_sample_indices(dataset: LoadedImageDataset) -> list[int] | None:
    if dataset.labels is None:
        return None
    return [int(dataset.labels[source_index]) for source_index in dataset.sample_indices]


@dataclass(slots=True)
class LoadedTextDataset:
    name: str
    root: Path
    entries: list[str]
    labels: list[int] | None
    item_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_entries(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class ImageDatasetSpec:
    name: str
    description: str
    download_mode: str
    supported_splits: tuple[str, ...]
    builder: ImageDatasetBuilder
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TextDatasetSpec:
    name: str
    description: str
    download_mode: str
    builder: TextDatasetBuilder
    notes: str = ""
