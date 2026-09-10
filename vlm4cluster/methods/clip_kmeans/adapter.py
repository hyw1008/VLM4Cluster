from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.clip_kmeans.core import run_clip_kmeans_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class CLIPKMeansMethod(ClusteringMethod):
    name = "clip_kmeans"
    description = "Direct K-Means baseline on shared OpenCLIP evaluation-split image features."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        image_step = (
            f"embed '{target_dataset}' target-test images with the shared OpenCLIP image encoder"
            if target_dataset is not None
            else "embed evaluation images with the shared OpenCLIP image encoder"
        )
        return [
            f"load OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            image_step,
            "run K-Means directly on the evaluation-split OpenCLIP features",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_clip_kmeans_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
