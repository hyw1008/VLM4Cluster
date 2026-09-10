from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from vlm4cluster.config import ExperimentConfig, FeatureExtractorConfig
from vlm4cluster.datasets import get_image_dataset_spec, get_text_dataset_spec, load_image_dataset
from vlm4cluster.evaluation import (
    INTERNAL_CLUSTERING_METRICS,
    evaluate_clustering,
    evaluate_internal_clustering,
    normalize_internal_evaluation_features,
)
from vlm4cluster.features import get_image_extractor, get_text_extractor
from vlm4cluster.features.base import FeatureSet
from vlm4cluster.methods import get_method
from vlm4cluster.methods.base import MethodInputs, MethodResult
from vlm4cluster.methods.checkpointing import is_full_imagenet_dataset
from vlm4cluster.methods.feature_cache import raw_image_embedding_cache_path
from vlm4cluster.models import get_openclip_spec
from vlm4cluster.utils.efficiency import collect_training_efficiency
from vlm4cluster.utils.io import ensure_dir, write_json
from vlm4cluster.utils.progress import get_progress_logger
from vlm4cluster.utils.reproducibility import set_seed


def _method_params_with_seed(params: dict[str, Any], seed: int) -> dict[str, Any]:
    seeded = dict(params)
    seeded.setdefault("seed", seed)
    seeded.setdefault("random_state", seed)
    return seeded


def _canonical_trial_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("Trial fingerprint values must contain only finite floats.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical_trial_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_trial_value(item) for item in value]
    raise TypeError(f"Unsupported trial fingerprint value type: {type(value).__name__}.")


def _prediction_config_payload(config: ExperimentConfig, method_params: dict[str, Any]) -> dict[str, Any]:
    return _canonical_trial_value(
        {
            "schema_version": 1,
            "seed": config.seed,
            "method": {"name": config.method.name, "params": method_params},
            "image_dataset": asdict(config.image_dataset),
            "text_dataset": asdict(config.text_dataset) if config.text_dataset is not None else None,
            "image_features": asdict(config.image_features) if config.image_features is not None else None,
            "text_features": asdict(config.text_features) if config.text_features is not None else None,
        }
    )


def _prediction_config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _effective_metric_names(config: ExperimentConfig) -> list[str]:
    if config.evaluation.internal_metrics_only:
        return list(INTERNAL_CLUSTERING_METRICS)
    return list(config.evaluation.metrics)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, project_root: str | Path | None = None) -> None:
        self.config = config
        method_cls = get_method(config.method.name)
        self.method_params = method_cls.resolve_params(
            _method_params_with_seed(config.method.params, config.seed)
        )
        self.project_root = Path(project_root or Path.cwd()).resolve()
        output_dir = Path(config.output_dir)
        self.base_output_dir = ensure_dir(
            output_dir if output_dir.is_absolute() else self.project_root / output_dir
        )
        self.prediction_config = _prediction_config_payload(config, self.method_params)
        self.trial_fingerprint = _prediction_config_fingerprint(self.prediction_config)
        self.trial_id = f"trial_{self.trial_fingerprint[:16]}"
        self.trial_dir = ensure_dir(self.base_output_dir / "trials" / self.trial_id)
        self.evaluation_mode = "internal" if config.evaluation.internal_metrics_only else "external"
        self.output_dir = ensure_dir(self.trial_dir / self.evaluation_mode)

    def build_plan(self) -> dict[str, Any]:
        image_spec = get_image_dataset_spec(self.config.image_dataset.name)
        text_spec = (
            get_text_dataset_spec(self.config.text_dataset.name) if self.config.text_dataset is not None else None
        )
        method_cls = get_method(self.config.method.name)
        method = method_cls()

        steps = [f"prepare image dataset '{image_spec.name}' ({image_spec.download_mode})"]
        if self.config.image_dataset.attack is not None and self.config.image_dataset.attack.enabled:
            attack = self.config.image_dataset.attack
            steps.append(
                f"materialize attack '{attack.name}' for splits {', '.join(attack.apply_to_splits)} "
                f"with decoder '{attack.decoder_path}'"
            )
        if method.requires_image_features:
            if self.config.image_features is None:
                raise ValueError(f"Method '{method.name}' requires [image_features].")
            steps.append(f"extract image features with '{self.config.image_features.name}'")
        if self.config.text_dataset is not None:
            steps.append(f"prepare text dataset '{text_spec.name}' ({text_spec.download_mode})")
        if method.requires_text_features:
            if self.config.text_features is None:
                raise ValueError(f"Method '{method.name}' requires [text_features].")
            steps.append(f"extract text features with '{self.config.text_features.name}'")
        steps.extend(method.plan_steps(self.method_params))
        if is_full_imagenet_dataset(self.config.image_dataset.name):
            steps.append(
                "measure train+eval time and run peak CPU/GPU memory, "
                "excluding AnyAttack image generation"
            )
        steps.append(f"run clustering method '{method.name}'")
        steps.append(f"evaluate with metrics: {', '.join(_effective_metric_names(self.config))}")

        return {
            "experiment": self.config.name,
            "base_output_dir": str(self.base_output_dir),
            "output_dir": str(self.output_dir),
            "trial_id": self.trial_id,
            "evaluation_mode": self.evaluation_mode,
            "image_dataset": {
                "name": image_spec.name,
                "download_mode": image_spec.download_mode,
                "supported_splits": list(image_spec.supported_splits),
                "notes": image_spec.notes,
            },
            "text_dataset": (
                {
                    "name": text_spec.name,
                    "download_mode": text_spec.download_mode,
                    "notes": text_spec.notes,
                }
                if text_spec is not None
                else None
            ),
            "method": {
                "name": method.name,
                "category": method.category,
                "requires_raw_images": method.requires_raw_images,
                "requires_image_features": method.requires_image_features,
                "requires_text_features": method.requires_text_features,
            },
            "steps": steps,
        }

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        progress = get_progress_logger("runner", self.method_params)
        progress.log(f"Starting experiment '{self.config.name}'")
        set_seed(self.config.seed)
        trial_manifest_path = self.trial_dir / "trial.json"
        write_json(
            trial_manifest_path,
            {
                "trial_id": self.trial_id,
                "fingerprint": f"sha256:{self.trial_fingerprint}",
                "fingerprint_schema": 1,
                "prediction_config": self.prediction_config,
            },
        )
        plan = self.build_plan()
        plan_path = self.output_dir / "plan.json"
        write_json(plan_path, plan)
        progress.log(f"Plan written to {plan_path}")

        if dry_run:
            return {"status": "dry-run", "plan_path": str(plan_path), "plan": plan}

        with collect_training_efficiency(
            enabled=is_full_imagenet_dataset(self.config.image_dataset.name),
            device=self.config.runtime.device,
        ) as training_efficiency:
            method_cls = get_method(self.config.method.name)
            method = method_cls()
            (
                result,
                image_dataset,
                evaluation_dataset,
                openclip_cache_artifact,
                prediction_artifact,
                metrics,
                extra_metrics,
            ) = self._execute_benchmark_run(method, progress)
        efficiency = training_efficiency.as_dict(
            checkpoint_loaded=bool(result.metadata.get("checkpoint_loaded", False))
        )
        effective_metrics = _effective_metric_names(self.config)
        silhouette_sample_size = self.config.evaluation.silhouette_sample_size
        actual_silhouette_sample_size = (
            None
            if not self.config.evaluation.internal_metrics_only
            else min(
                len(result.predictions),
                len(result.predictions) if silhouette_sample_size is None else silhouette_sample_size,
            )
        )
        report = {
            "status": "completed",
            "experiment": self.config.name,
            "seed": self.config.seed,
            "method": result.method_name,
            "evaluation_split": evaluation_dataset.split,
            "metrics": metrics,
            "extra_metrics": extra_metrics,
            "efficiency": efficiency,
            "image_dataset_metadata": image_dataset.metadata,
            "evaluation_metadata": {
                "mode": self.evaluation_mode,
                "metrics": effective_metrics,
                "uses_ground_truth_labels": not self.config.evaluation.internal_metrics_only,
                "dataset": evaluation_dataset.name,
                "split": evaluation_dataset.split,
                "sample_count": len(result.predictions),
                "feature_space": (
                    "l2_normalized_common_openclip_image"
                    if self.config.evaluation.internal_metrics_only
                    else None
                ),
                "silhouette_metric": (
                    self.config.evaluation.silhouette_metric
                    if self.config.evaluation.internal_metrics_only
                    else None
                ),
                "silhouette_sample_size": actual_silhouette_sample_size,
            },
            "trial": {
                "id": self.trial_id,
                "fingerprint": f"sha256:{self.trial_fingerprint}",
                "fingerprint_schema": 1,
                "prediction_config": self.prediction_config,
            },
            "artifacts": {
                "base_output_dir": str(self.base_output_dir),
                "output_dir": str(self.output_dir),
                "plan_path": str(plan_path),
                "trial_manifest_path": str(trial_manifest_path),
                "predictions": prediction_artifact,
                "openclip_cache": openclip_cache_artifact,
            },
            "result_metadata": result.metadata,
        }
        write_json(self.output_dir / "report.json", report)
        progress.log(f"Report written to {self.output_dir / 'report.json'}")
        return report

    def _execute_benchmark_run(self, method, progress):
        with progress.stage(f"Preparing image dataset '{self.config.image_dataset.name}'"):
            image_dataset = load_image_dataset(self.config.image_dataset)
        text_dataset = None
        if self.config.text_dataset is not None:
            with progress.stage(f"Preparing text dataset '{self.config.text_dataset.name}'"):
                text_dataset = get_text_dataset_spec(self.config.text_dataset.name).builder(
                    self.config.text_dataset,
                    image_dataset.class_names,
                )

        image_features = None
        if method.requires_image_features:
            if self.config.image_features is None:
                raise ValueError(f"Method '{method.name}' requires [image_features].")
            with progress.stage(f"Extracting image features with '{self.config.image_features.name}'"):
                image_features = self._extract_image_features(
                    image_dataset,
                    self.config.image_features,
                )

        text_features = None
        if method.requires_text_features:
            if text_dataset is None:
                raise ValueError(f"Method '{method.name}' requires a text dataset.")
            if self.config.text_features is None:
                raise ValueError(f"Method '{method.name}' requires [text_features].")
            with progress.stage(f"Extracting text features with '{self.config.text_features.name}'"):
                text_features = self._extract_text_features(
                    text_dataset,
                    self.config.text_features,
                )

        with progress.stage(f"Running method '{method.name}'"):
            result = method.run(
                MethodInputs(
                    image_dataset=image_dataset,
                    text_dataset=text_dataset,
                    image_features=image_features,
                    text_features=text_features,
                    image_dataset_config=self.config.image_dataset,
                    text_dataset_config=self.config.text_dataset,
                    runtime_config=self.config.runtime,
                    output_dir=self.output_dir,
                ),
                self.method_params,
            )

        extra_predictions = _extra_predictions(result.metadata)
        with progress.stage("Saving clustering predictions"):
            evaluation_dataset = self._load_result_evaluation_dataset(image_dataset, result)
            openclip_cache_artifact = self._openclip_cache_artifact(evaluation_dataset, result)
            prediction_artifact = _save_prediction_artifact(
                self.output_dir / "predictions.npz",
                result.predictions,
                extra_predictions,
                evaluation_dataset.sample_indices,
            )

        with progress.stage("Evaluating clustering metrics"):
            if self.config.evaluation.internal_metrics_only:
                if openclip_cache_artifact is None:
                    raise ValueError(
                        f"Method '{result.method_name}' did not report the OpenCLIP model "
                        "needed for internal metrics."
                    )
                normalized_features, sample_order_verified = _load_internal_evaluation_features(
                    Path(openclip_cache_artifact["path"]),
                    expected_sample_indices=evaluation_dataset.sample_indices,
                )
                openclip_cache_artifact["feature_dim"] = int(normalized_features.shape[1])
                openclip_cache_artifact["sample_order_verified"] = sample_order_verified
                metrics = evaluate_internal_clustering(
                    normalized_features,
                    result.predictions,
                    silhouette_metric=self.config.evaluation.silhouette_metric,
                    silhouette_sample_size=self.config.evaluation.silhouette_sample_size,
                    random_state=self.config.seed,
                )
                extra_metrics = {
                    name: evaluate_internal_clustering(
                        normalized_features,
                        predictions,
                        silhouette_metric=self.config.evaluation.silhouette_metric,
                        silhouette_sample_size=self.config.evaluation.silhouette_sample_size,
                        random_state=self.config.seed,
                    )
                    for name, predictions in extra_predictions.items()
                }
            else:
                labels = result.evaluation_labels
                if labels is None and evaluation_dataset.labels is not None:
                    labels = [
                        evaluation_dataset.labels[index]
                        for index in evaluation_dataset.sample_indices
                    ]
                metrics = evaluate_clustering(
                    labels,
                    result.predictions,
                    self.config.evaluation.metrics,
                )
                extra_metrics = {
                    name: evaluate_clustering(
                        labels,
                        predictions,
                        self.config.evaluation.metrics,
                    )
                    for name, predictions in extra_predictions.items()
                }

        return (
            result,
            image_dataset,
            evaluation_dataset,
            openclip_cache_artifact,
            prediction_artifact,
            metrics,
            extra_metrics,
        )

    def _load_result_evaluation_dataset(self, initial_dataset, result: MethodResult):
        metadata = result.metadata
        evaluation_name = str(metadata.get("evaluation_dataset", initial_dataset.name))
        evaluation_split = str(
            result.evaluation_split
            or metadata.get("evaluation_split")
            or metadata.get("test_split")
            or initial_dataset.split
        )
        max_samples = self.method_params.get("max_test_samples", self.config.image_dataset.max_samples)
        same_dataset = _normalized_dataset_name(evaluation_name) == _normalized_dataset_name(
            self.config.image_dataset.name
        )
        evaluation_config = replace(
            self.config.image_dataset,
            name=evaluation_name,
            split=evaluation_split,
            download=self.config.image_dataset.download if same_dataset else None,
            transform_preset="none",
            max_samples=max_samples,
        )
        if (
            _normalized_dataset_name(initial_dataset.name) == _normalized_dataset_name(evaluation_name)
            and initial_dataset.split == evaluation_split
            and initial_dataset.num_samples == len(result.predictions)
        ):
            return initial_dataset
        evaluation_dataset = load_image_dataset(evaluation_config)
        if evaluation_dataset.num_samples != len(result.predictions):
            raise ValueError(
                "Prediction artifact cannot be aligned with the evaluation dataset: "
                f"{len(result.predictions)} prediction(s) for "
                f"{evaluation_dataset.num_samples} sample(s) in "
                f"{evaluation_dataset.name}/{evaluation_dataset.split}."
            )
        return evaluation_dataset

    def _openclip_cache_artifact(self, evaluation_dataset, result: MethodResult) -> dict[str, Any] | None:
        metadata = result.metadata
        pretraining = metadata.get("openclip_pretraining", self.method_params.get("openclip_pretraining"))
        backbone = metadata.get("openclip_backbone", self.method_params.get("openclip_backbone"))
        if pretraining is None or backbone is None:
            return None
        spec = get_openclip_spec(str(pretraining), str(backbone))
        cache_path = raw_image_embedding_cache_path(evaluation_dataset, spec.cache_key)
        return {
            "path": str(cache_path),
            "format": "numpy-npz",
            "features_key": "features",
            "dataset": evaluation_dataset.name,
            "split": evaluation_dataset.split,
            "pretraining": spec.benchmark_pretraining,
            "backbone": spec.benchmark_backbone,
            "cache_key": spec.cache_key,
            "sample_count": evaluation_dataset.num_samples,
            "sample_order_sha256": _array_sha256(evaluation_dataset.sample_indices),
            "l2_normalized_for_evaluation": self.config.evaluation.internal_metrics_only,
            "exists": cache_path.exists(),
        }

    def _extract_image_features(self, dataset, config: FeatureExtractorConfig | None):
        if config is None:
            raise ValueError("image_features config is required.")
        cache_path = self._feature_cache_path(
            modality="image",
            dataset_name=dataset.name,
            split=dataset.split,
            extractor_name=config.name,
        )
        if config.cache and cache_path.exists() and not config.force_recompute:
            return self._load_feature_cache(cache_path)
        extractor_cls = get_image_extractor(config.name)
        extractor = extractor_cls()
        feature_set = extractor.extract(dataset, config, self.config.runtime, self.output_dir)
        if config.cache:
            self._save_feature_cache(cache_path, feature_set)
            feature_set.metadata["cache_path"] = str(cache_path)
        return feature_set

    def _extract_text_features(self, dataset, config: FeatureExtractorConfig | None):
        if config is None:
            raise ValueError("text_features config is required.")
        cache_path = self._feature_cache_path(
            modality="text",
            dataset_name=dataset.name,
            split="corpus",
            extractor_name=config.name,
        )
        if config.cache and cache_path.exists() and not config.force_recompute:
            return self._load_feature_cache(cache_path)
        extractor_cls = get_text_extractor(config.name)
        extractor = extractor_cls()
        feature_set = extractor.extract(dataset, config, self.config.runtime, self.output_dir)
        if config.cache:
            self._save_feature_cache(cache_path, feature_set)
            feature_set.metadata["cache_path"] = str(cache_path)
        return feature_set

    def _feature_cache_path(self, *, modality: str, dataset_name: str, split: str, extractor_name: str) -> Path:
        cache_dir = ensure_dir(self.base_output_dir / "feature_cache" / modality)
        return cache_dir / f"{dataset_name}__{split}__{extractor_name}.npz"

    def _save_feature_cache(self, path: Path, feature_set: FeatureSet) -> None:
        import numpy as np

        labels = feature_set.labels if feature_set.labels is not None else []
        np.savez_compressed(
            path,
            name=np.asarray([feature_set.name], dtype=object),
            modality=np.asarray([feature_set.modality], dtype=object),
            matrix=feature_set.matrix,
            labels=np.asarray(labels, dtype=int),
            has_labels=np.asarray([feature_set.labels is not None], dtype=bool),
            item_ids=np.asarray(feature_set.item_ids, dtype=object),
            metadata=np.asarray([json.dumps(feature_set.metadata)], dtype=object),
        )

    def _load_feature_cache(self, path: Path) -> FeatureSet:
        import numpy as np

        payload = np.load(path, allow_pickle=True)
        has_labels = bool(payload["has_labels"][0])
        labels = payload["labels"].tolist() if has_labels else None
        item_ids = [str(item) for item in payload["item_ids"].tolist()]
        metadata = json.loads(str(payload["metadata"][0]))
        metadata["cache_path"] = str(path)
        return FeatureSet(
            name=str(payload["name"][0]),
            modality=str(payload["modality"][0]),
            matrix=payload["matrix"],
            labels=labels,
            item_ids=item_ids,
            metadata=metadata,
        )


def _normalized_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_prediction_array(predictions, *, name: str):
    import numpy as np

    array = np.asarray(predictions)
    if array.ndim != 1:
        raise ValueError(f"Prediction group '{name}' must be one-dimensional, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"Prediction group '{name}' cannot be empty.")
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"Prediction group '{name}' must contain integer cluster IDs, got dtype {array.dtype}.")
    return array.astype(np.int64, copy=False)


def _array_sha256(values) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _save_prediction_artifact(
    path: Path,
    predictions: list[int],
    extra_predictions: dict[str, list[int]],
    sample_indices: list[int],
) -> dict[str, Any]:
    import numpy as np

    primary = _coerce_prediction_array(predictions, name="main")
    sample_index_array = np.asarray(sample_indices, dtype=np.int64)
    if sample_index_array.ndim != 1 or sample_index_array.shape[0] != primary.shape[0]:
        raise ValueError(
            "Prediction sample order is misaligned: "
            f"{primary.shape[0]} prediction(s) and {sample_index_array.shape[0]} sample index value(s)."
        )

    arrays: dict[str, Any] = {"main": primary, "sample_indices": sample_index_array}
    extras_artifact: dict[str, dict[str, Any]] = {}
    for index, (name, values) in enumerate(sorted(extra_predictions.items())):
        array = _coerce_prediction_array(values, name=name)
        if array.shape[0] != primary.shape[0]:
            raise ValueError(
                f"Extra prediction group '{name}' has {array.shape[0]} row(s); "
                f"the main prediction has {primary.shape[0]}."
            )
        key = f"extra_{index:04d}"
        arrays[key] = array
        extras_artifact[name] = {
            "key": key,
            "num_samples": int(array.shape[0]),
            "dtype": str(array.dtype),
            "content_sha256": _array_sha256(array),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(path)
    return {
        "path": str(path),
        "format": "numpy-npz",
        "primary": {
            "key": "main",
            "num_samples": int(primary.shape[0]),
            "num_clusters": int(np.unique(primary).size),
            "dtype": str(primary.dtype),
            "content_sha256": _array_sha256(primary),
        },
        "sample_indices": {
            "key": "sample_indices",
            "content_sha256": _array_sha256(sample_index_array),
        },
        "extras": extras_artifact,
    }


def _load_internal_evaluation_features(
    path: Path,
    *,
    expected_sample_indices: list[int],
):
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(
            "Internal metrics require the shared OpenCLIP evaluation feature cache, "
            f"but it was not found: {path}"
        )
    with np.load(path, allow_pickle=False) as payload:
        if "features" not in payload.files:
            raise KeyError(f"OpenCLIP cache does not contain a 'features' array: {path}")
        features = np.asarray(payload["features"], dtype=np.float32)
        cached_sample_indices = (
            np.asarray(payload["sample_indices"], dtype=np.int64)
            if "sample_indices" in payload.files
            else None
        )

    expected = np.asarray(expected_sample_indices, dtype=np.int64)
    if features.shape[0] != expected.shape[0]:
        raise ValueError(
            "OpenCLIP cache and prediction rows are misaligned: "
            f"{features.shape[0]} cached feature row(s) and {expected.shape[0]} expected sample(s)."
        )
    if cached_sample_indices is not None and not np.array_equal(cached_sample_indices, expected):
        raise ValueError(f"OpenCLIP cache sample order does not match the prediction artifact: {path}")
    normalized = normalize_internal_evaluation_features(features)
    return normalized, cached_sample_indices is not None


def _extra_predictions(metadata: dict[str, Any]) -> dict[str, list[int]]:
    raw_predictions = metadata.get("extra_predictions", {})
    if not isinstance(raw_predictions, dict):
        return {}
    return {
        str(name): [int(prediction) for prediction in predictions]
        for name, predictions in raw_predictions.items()
        if isinstance(predictions, list)
    }
