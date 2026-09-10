from __future__ import annotations

from typing import Type

from vlm4cluster.features.base import ImageFeatureExtractor, TextFeatureExtractor


IMAGE_EXTRACTORS: dict[str, Type[ImageFeatureExtractor]] = {}
TEXT_EXTRACTORS: dict[str, Type[TextFeatureExtractor]] = {}


def register_image_extractor(cls: Type[ImageFeatureExtractor]) -> Type[ImageFeatureExtractor]:
    IMAGE_EXTRACTORS[cls.name] = cls
    return cls


def register_text_extractor(cls: Type[TextFeatureExtractor]) -> Type[TextFeatureExtractor]:
    TEXT_EXTRACTORS[cls.name] = cls
    return cls


def get_image_extractor(name: str) -> Type[ImageFeatureExtractor]:
    try:
        return IMAGE_EXTRACTORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown image feature extractor '{name}'.") from exc


def get_text_extractor(name: str) -> Type[TextFeatureExtractor]:
    try:
        return TEXT_EXTRACTORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown text feature extractor '{name}'.") from exc


def list_image_extractors() -> list[Type[ImageFeatureExtractor]]:
    return [IMAGE_EXTRACTORS[key] for key in sorted(IMAGE_EXTRACTORS)]


def list_text_extractors() -> list[Type[TextFeatureExtractor]]:
    return [TEXT_EXTRACTORS[key] for key in sorted(TEXT_EXTRACTORS)]
