from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.magic.core import _resolve_prediction_head, run_magic_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class MAGICMethod(ClusteringMethod):
    name = "magic"
    description = (
        "Multi-granularity language-informed image clustering with MAGIC training and TAC-style text construction."
    )
    category = "image_text"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    @classmethod
    def resolve_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        resolved = super().resolve_params(params)
        resolved["prediction_head"] = _resolve_prediction_head(resolved)
        resolved.pop("head", None)
        return resolved

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        source_dataset = str(params.get("source_dataset_name", "source dataset"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        prediction_head = _resolve_prediction_head(params)
        prediction_step = {
            "image": "run the trained MAGIC image cluster head on the evaluation split",
            "text": "run the trained MAGIC text cluster head on the evaluation split",
            "both": "run the trained MAGIC image and text cluster heads on the evaluation split",
        }[prediction_head]
        image_step = (
            f"embed '{source_dataset}' source-train images and '{target_dataset}' target-test images with the shared OpenCLIP image encoder"
            if target_dataset is not None
            else "embed source train and evaluation split images with the shared OpenCLIP image encoder"
        )
        return [
            f"load OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            image_step,
            "load shared WordNetNouns.csv and select TAC-style discriminative nouns",
            "construct MAGIC fine/coarse text granularities from TAC-selected nouns instead of Qwen-generated descriptions",
            "mine image-space and text-space train neighbors",
            "train MAGIC cross-granularity attention, semantic adapters, and image/text cluster heads",
            prediction_step,
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_magic_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
