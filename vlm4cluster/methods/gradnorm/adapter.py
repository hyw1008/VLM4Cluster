from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.gradnorm.core import run_gradnorm_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class GradNormMethod(ClusteringMethod):
    name = "gradnorm"
    description = "Gradient-based noun filtering for language-assisted image clustering with a no-train TAC-style pipeline."
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
            else "embed train/test images with the shared OpenCLIP image encoder"
        )
        return [
            f"load OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            "load the shared WordNetNouns.csv resource from the data directory and encode its noun pool with the shared OpenCLIP text encoder",
            image_step,
            "filter nouns with the GradNorm scoring rule on train-image proxy clusters",
            "retrieve text counterparts for test images and run the no-train concat-kmeans variant",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_gradnorm_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
