from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm4cluster.config import FeatureExtractorConfig, RuntimeConfig
from vlm4cluster.datasets.base import LoadedImageDataset, LoadedTextDataset


@dataclass(slots=True)
class FeatureSet:
    name: str
    modality: str
    matrix: Any
    labels: list[int] | None
    item_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)


class ImageFeatureExtractor(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def extract(
        self,
        dataset: LoadedImageDataset,
        config: FeatureExtractorConfig,
        runtime: RuntimeConfig,
        output_dir: Path,
    ) -> FeatureSet:
        raise NotImplementedError


class TextFeatureExtractor(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def extract(
        self,
        dataset: LoadedTextDataset,
        config: FeatureExtractorConfig,
        runtime: RuntimeConfig,
        output_dir: Path,
    ) -> FeatureSet:
        raise NotImplementedError
