from __future__ import annotations

from vlm4cluster.utils.deps import require_module


def build_transform(preset: str):
    if preset == "none":
        return None

    transforms = require_module("torchvision.transforms", "pip install torch torchvision")

    if preset == "standard_train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    if preset == "simclr":
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    raise ValueError(f"Unknown transform preset '{preset}'.")
