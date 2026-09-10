from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_ANYATTACK_EPS = 8.0 / 255.0


@dataclass(slots=True)
class DatasetAttackConfig:
    name: str
    enabled: bool = True
    apply_to_splits: list[str] = field(default_factory=lambda: ["test", "val"])
    decoder_path: str = "data/anyattack/checkpoints/coco_cos.pt"
    variant: str = "anyattack_cos"
    eps: float = DEFAULT_ANYATTACK_EPS
    target_strategy: str = "shuffle"
    avoid_same_label: bool = True
    batch_size: int = 128
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    name: str
    root: str = "data"
    split: str = "train"
    download: bool | None = None
    transform_preset: str = "none"
    max_samples: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    attack: DatasetAttackConfig | None = None


@dataclass(slots=True)
class FeatureExtractorConfig:
    name: str
    cache: bool = True
    force_recompute: bool = False
    batch_size: int = 128
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MethodConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationConfig:
    metrics: list[str] = field(default_factory=lambda: ["nmi", "ari", "acc"])
    internal_metrics_only: bool = False
    silhouette_metric: str = "cosine"
    silhouette_sample_size: int | None = 5000


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cpu"
    num_workers: int = 0


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    output_dir: str = ""
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    image_dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="cifar10"))
    text_dataset: DatasetConfig | None = None
    image_features: FeatureExtractorConfig | None = None
    text_features: FeatureExtractorConfig | None = None
    method: MethodConfig = field(default_factory=lambda: MethodConfig(name=""))
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


_KNOWN_TOP_LEVEL_KEYS = {
    "name",
    "seed",
    "output_dir",
    "runtime",
    "image_dataset",
    "text_dataset",
    "image_features",
    "text_features",
    "method",
    "evaluation",
}

_OVERRIDE_ALIASES = {
    "dataset": "image_dataset.name",
    "split": "image_dataset.split",
    "attack": "image_dataset.attack.name",
    "adversarial": "image_dataset.attack.enabled",
    "method": "method.name",
    "device": "runtime.device",
    "num_workers": "runtime.num_workers",
    "metrics": "evaluation.metrics",
    "internal_metrics_only": "evaluation.internal_metrics_only",
}


def _path_token(value: Any) -> str:
    token = str(value).strip().lower()
    for old, new in (
        ("/", "-"),
        ("\\", "-"),
        (" ", "-"),
        ("_", "-"),
    ):
        token = token.replace(old, new)
    return "-".join(part for part in token.split("-") if part)


def _attack_output_token(attack: dict[str, Any]) -> str | None:
    raw_name = attack.get("name", "")
    if raw_name is True:
        raw_name = "anyattack"
    name = str(raw_name).strip()
    if not name:
        return None
    if attack.get("enabled") is False:
        return None

    checkpoint = Path(str(attack.get("decoder_path", "coco_cos.pt"))).stem
    variant = str(attack.get("variant", "anyattack_cos")).strip()
    eps = f"{int(round(float(attack.get('eps', DEFAULT_ANYATTACK_EPS)) * 255))}-255"
    seed = str(attack.get("seed", 42)).strip()
    return _path_token(f"attack-{name}-{variant}-{checkpoint}-eps{eps}-seed{seed}")


def build_default_output_dir(raw: dict[str, Any], *, seed: int, config_path: Path) -> str:
    image_dataset = _as_dict(raw.get("image_dataset"))
    method = _as_dict(raw.get("method"))
    method_params = _as_dict(method.get("params"))

    if not image_dataset or not method:
        return f"outputs/{config_path.stem}"

    method_name = _path_token(method["name"])
    dataset_name = _path_token(image_dataset["name"])
    pretraining = _path_token(method_params.get("openclip_pretraining", "LAION400M"))
    backbone = _path_token(method_params.get("openclip_backbone", "ViT-B/32"))
    attack_token = _attack_output_token(_as_dict(image_dataset.get("attack")))
    if attack_token is not None:
        dataset_name = f"{dataset_name}/{attack_token}"
    return f"outputs/{method_name}/{dataset_name}/{pretraining}_{backbone}/seed_{seed}"


def _resolve_output_dir(raw: dict[str, Any], *, seed: int, config_path: Path) -> str:
    configured = raw.get("output_dir")
    if configured is None:
        return build_default_output_dir(raw, seed=seed, config_path=config_path)
    configured_text = str(configured).strip()
    if configured_text == "" or configured_text.lower() == "auto":
        return build_default_output_dir(raw, seed=seed, config_path=config_path)
    return configured_text


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a TOML table, got {type(value)!r}")
    return dict(value)


def _as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"Config value '{key}' must be a boolean, got {type(value).__name__}.")


def _parse_override_value(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"none", "null"}:
        return None

    try:
        return tomllib.loads(f"value = {stripped}")["value"]
    except tomllib.TOMLDecodeError:
        pass

    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",")]

    if "," in stripped:
        return [item.strip().strip("'\"") for item in stripped.split(",")]

    return stripped


def _normalize_override_key(key: str) -> str:
    key = key.strip()
    if not key:
        raise ValueError("Override key cannot be empty.")
    if key in _OVERRIDE_ALIASES:
        return _OVERRIDE_ALIASES[key]
    if key.startswith("dataset."):
        return f"image_dataset.{key.removeprefix('dataset.')}"
    if key.startswith("attack."):
        return f"image_dataset.attack.{key.removeprefix('attack.')}"
    if key.startswith("method."):
        suffix = key.removeprefix("method.")
        if suffix == "name" or suffix.startswith("params."):
            return key
        return f"method.params.{suffix}"
    return key


def _set_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if parts[0] not in _KNOWN_TOP_LEVEL_KEYS:
        known = ", ".join(sorted(_KNOWN_TOP_LEVEL_KEYS | set(_OVERRIDE_ALIASES)))
        raise ValueError(f"Unknown override root '{parts[0]}' in '{dotted_key}'. Known roots/aliases: {known}")

    cursor = target
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise TypeError(f"Cannot set nested override '{dotted_key}': '{part}' is not a table.")
        cursor = current
    cursor[parts[-1]] = value


def parse_config_overrides(items: list[str] | tuple[str, ...]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        normalized_key = _normalize_override_key(key)
        _set_nested_value(overrides, normalized_key, _parse_override_value(raw_value))
    return overrides


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_attack_config(data: dict[str, Any], *, default_seed: int) -> DatasetAttackConfig | None:
    if not data:
        return None

    name = data.get("name")
    if name is True:
        name = "anyattack"
    if name is None or str(name).strip().lower() in {"", "none", "null", "false"}:
        return None

    params = _as_dict(data.get("params"))
    apply_to_splits_raw = data.get("apply_to_splits", ["test", "val"])
    if isinstance(apply_to_splits_raw, str):
        apply_to_splits = [part.strip() for part in apply_to_splits_raw.split(",") if part.strip()]
    else:
        apply_to_splits = [str(split) for split in apply_to_splits_raw]

    return DatasetAttackConfig(
        name=str(name),
        enabled=bool(data.get("enabled", True)),
        apply_to_splits=apply_to_splits,
        decoder_path=str(data.get("decoder_path", "data/anyattack/checkpoints/coco_cos.pt")),
        variant=str(data.get("variant", "anyattack_cos")),
        eps=float(data.get("eps", DEFAULT_ANYATTACK_EPS)),
        target_strategy=str(data.get("target_strategy", "shuffle")),
        avoid_same_label=bool(data.get("avoid_same_label", True)),
        batch_size=int(data.get("batch_size", 128)),
        seed=int(data.get("seed", default_seed)),
        params=params,
    )


def _load_dataset_config(
    data: dict[str, Any],
    *,
    default_split: str = "train",
    default_attack_seed: int = 42,
) -> DatasetConfig:
    params = _as_dict(data.get("params"))
    attack = _load_attack_config(_as_dict(data.get("attack")), default_seed=default_attack_seed)
    return DatasetConfig(
        name=str(data["name"]),
        root=str(data.get("root", "data")),
        split=str(data.get("split", default_split)),
        download=data.get("download"),
        transform_preset=str(data.get("transform_preset", "none")),
        max_samples=data.get("max_samples"),
        params=params,
        attack=attack,
    )


def _load_feature_config(data: dict[str, Any]) -> FeatureExtractorConfig:
    params = _as_dict(data.get("params"))
    return FeatureExtractorConfig(
        name=str(data["name"]),
        cache=bool(data.get("cache", True)),
        force_recompute=bool(data.get("force_recompute", False)),
        batch_size=int(data.get("batch_size", 128)),
        params=params,
    )


def load_experiment_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if overrides:
        raw = _deep_merge_dicts(raw, overrides)

    image_dataset = _as_dict(raw.get("image_dataset"))
    if not image_dataset:
        raise ValueError("Config must define [image_dataset].")

    method = _as_dict(raw.get("method"))
    if not method:
        raise ValueError("Config must define [method].")

    runtime_data = _as_dict(raw.get("runtime"))
    evaluation_data = _as_dict(raw.get("evaluation"))
    text_dataset_data = _as_dict(raw.get("text_dataset"))
    image_features_data = _as_dict(raw.get("image_features"))
    text_features_data = _as_dict(raw.get("text_features"))

    seed = int(raw.get("seed", 42))
    silhouette_sample_size_raw = evaluation_data.get("silhouette_sample_size", 5000)
    config = ExperimentConfig(
        name=str(raw.get("name", config_path.stem)),
        seed=seed,
        output_dir=_resolve_output_dir(raw, seed=seed, config_path=config_path),
        runtime=RuntimeConfig(
            device=str(runtime_data.get("device", "cpu")),
            num_workers=int(runtime_data.get("num_workers", 0)),
        ),
        image_dataset=_load_dataset_config(image_dataset, default_attack_seed=seed),
        text_dataset=_load_dataset_config(text_dataset_data, default_attack_seed=seed) if text_dataset_data else None,
        image_features=_load_feature_config(image_features_data) if image_features_data else None,
        text_features=_load_feature_config(text_features_data) if text_features_data else None,
        method=MethodConfig(
            name=str(method["name"]),
            params=_as_dict(method.get("params")),
        ),
        evaluation=EvaluationConfig(
            metrics=[str(metric) for metric in evaluation_data.get("metrics", ["nmi", "ari", "acc"])],
            internal_metrics_only=_as_bool(
                evaluation_data.get("internal_metrics_only", False),
                key="evaluation.internal_metrics_only",
            ),
            silhouette_metric=str(evaluation_data.get("silhouette_metric", "cosine")),
            silhouette_sample_size=(
                None if silhouette_sample_size_raw is None else int(silhouette_sample_size_raw)
            ),
        ),
    )
    return config
