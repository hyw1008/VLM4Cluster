from __future__ import annotations

from dataclasses import replace

from vlm4cluster.config import DatasetConfig
from vlm4cluster.datasets.base import LoadedImageDataset


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def attack_applies_to_split(config: DatasetConfig) -> bool:
    attack = config.attack
    if attack is None or not attack.enabled:
        return False
    split = config.split.strip().lower()
    return split in {value.strip().lower() for value in attack.apply_to_splits}


def raw_dataset_config_for_attack(config: DatasetConfig) -> DatasetConfig:
    return replace(config, transform_preset="none")


def apply_image_attack(dataset: LoadedImageDataset, config: DatasetConfig) -> LoadedImageDataset:
    attack = config.attack
    if attack is None or not attack.enabled:
        return dataset
    if not attack_applies_to_split(config):
        return dataset

    normalized = _normalize_name(attack.name)
    if normalized != "anyattack":
        raise ValueError(f"Unknown image attack '{attack.name}'. Supported attacks: anyattack")

    from vlm4cluster.attacks.anyattack import materialize_anyattack_dataset

    return materialize_anyattack_dataset(dataset, config, attack)
