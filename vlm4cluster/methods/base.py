from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm4cluster.datasets.base import LoadedImageDataset, LoadedTextDataset
from vlm4cluster.features.base import FeatureSet

if TYPE_CHECKING:
    from vlm4cluster.config import DatasetConfig, RuntimeConfig


@dataclass(slots=True)
class MethodInputs:
    image_dataset: LoadedImageDataset
    text_dataset: LoadedTextDataset | None
    image_features: FeatureSet | None
    text_features: FeatureSet | None
    image_dataset_config: "DatasetConfig"
    text_dataset_config: "DatasetConfig | None"
    runtime_config: "RuntimeConfig"
    output_dir: Path


@dataclass(slots=True)
class MethodResult:
    method_name: str
    predictions: list[int]
    evaluation_labels: list[int] | None = None
    evaluation_split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ClusteringMethod(ABC):
    name: str = ""
    description: str = ""
    category: str = "image_only"
    requires_raw_images: bool = False
    requires_image_features: bool = True
    requires_text_features: bool = False

    @classmethod
    def resolve_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        return []

    @abstractmethod
    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        raise NotImplementedError
