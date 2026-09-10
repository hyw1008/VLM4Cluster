from __future__ import annotations

import csv
from pathlib import Path

from vlm4cluster.config import DatasetConfig
from vlm4cluster.datasets.base import LoadedTextDataset, TextDatasetSpec
from vlm4cluster.datasets.registry import register_text_dataset


def _normalize_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ")


def _build_class_name_corpus(config: DatasetConfig, class_names: list[str]) -> LoadedTextDataset:
    entries = [_normalize_label(class_name) for class_name in class_names]
    labels = list(range(len(entries)))
    item_ids = [f"class-{index}" for index in labels]
    return LoadedTextDataset(
        name="class_names",
        root=Path(config.root).expanduser() / "class_names",
        entries=entries,
        labels=labels,
        item_ids=item_ids,
    )


def _resolve_shared_wordnet_csv(config: DatasetConfig) -> Path:
    explicit = config.params.get("wordnet_csv") or config.params.get("wordnet_path")
    if explicit is not None:
        candidate = Path(str(explicit)).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Shared WordNetNouns.csv not found: {candidate}")

    candidate = Path(config.root).expanduser() / "WordNetNouns.csv"
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        "Shared WordNetNouns.csv dataset not found. "
        f"Please place it at {candidate} or specify text_dataset.params.wordnet_csv."
    )


def _build_shared_wordnet_dataset(config: DatasetConfig, class_names: list[str]) -> LoadedTextDataset:
    del class_names
    csv_path = _resolve_shared_wordnet_csv(config)
    entries: list[str] = []
    item_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "word" not in reader.fieldnames:
            raise ValueError(f"Invalid WordNetNouns.csv format: {csv_path}")
        for index, row in enumerate(reader):
            word = str(row["word"]).strip()
            if not word:
                continue
            entries.append(word)
            item_ids.append(f"wordnet-{index}")

    return LoadedTextDataset(
        name="wordnet",
        root=csv_path,
        entries=entries,
        labels=None,
        item_ids=item_ids,
        metadata={"source_csv": str(csv_path), "num_entries": len(entries)},
    )


register_text_dataset(
    TextDatasetSpec(
        name="class_names",
        description="Use image class names directly as text entries.",
        download_mode="generated",
        builder=_build_class_name_corpus,
    )
)

register_text_dataset(
    TextDatasetSpec(
        name="wordnet",
        description="Shared WordNetNouns.csv dataset for language-assisted methods.",
        download_mode="manual",
        builder=_build_shared_wordnet_dataset,
        notes="Expected at data/WordNetNouns.csv unless overridden by text_dataset.params.wordnet_csv.",
    )
)
