from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.sac.core import run_sac_pipeline


@register_method
class SACMethod(ClusteringMethod):
    name = "sac"
    description = (
        "Semantic-Augmented Image Clustering with SAC adaptive collaboration training and TAC-style text construction."
    )
    category = "image_text"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        source_dataset = str(params.get("source_dataset_name", "source dataset"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        image_step = (
            f"embed '{source_dataset}' source-train images and '{target_dataset}' target-test images with the shared OpenCLIP image encoder"
            if target_dataset is not None
            else "embed source train and evaluation split images with the shared OpenCLIP image encoder"
        )
        return [
            f"load OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            image_step,
            "load shared WordNetNouns.csv and build text counterparts with TAC noun filtering and retrieval",
            "mine image-space and text-space train neighbors",
            "train SAC image/text cluster heads with adaptive robust contrastive, alignment, and balance losses",
            "run the trained SAC image cluster head on the evaluation split",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_sac_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
