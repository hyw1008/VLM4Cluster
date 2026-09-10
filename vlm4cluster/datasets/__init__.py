from vlm4cluster.config import DatasetConfig
from vlm4cluster.attacks import apply_image_attack, attack_applies_to_split, raw_dataset_config_for_attack
from vlm4cluster.datasets import image_datasets as _image_datasets
from vlm4cluster.datasets import text_datasets as _text_datasets
from vlm4cluster.datasets.registry import (
    get_image_dataset_spec,
    get_text_dataset_spec,
    list_image_dataset_specs,
    list_text_dataset_specs,
)

__all__ = [
    "get_image_dataset_spec",
    "get_text_dataset_spec",
    "list_image_dataset_specs",
    "list_text_dataset_specs",
    "load_image_dataset",
    "load_text_dataset",
]


def load_image_dataset(config: DatasetConfig):
    spec = get_image_dataset_spec(config.name)
    if attack_applies_to_split(config):
        dataset = spec.builder(raw_dataset_config_for_attack(config))
        return apply_image_attack(dataset, config)
    return spec.builder(config)


def load_text_dataset(config: DatasetConfig, class_names: list[str]):
    return get_text_dataset_spec(config.name).builder(config, class_names)
