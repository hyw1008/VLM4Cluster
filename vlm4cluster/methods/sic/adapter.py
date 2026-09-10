from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.sic.core import run_sic_pipeline


@register_method
class SICMethod(ClusteringMethod):
    name = "sic"
    description = "Semantic-Enhanced Image Clustering with OpenCLIP features and fixed noun-space construction."
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
            "load the shared WordNetNouns.csv resource and encode SIC nouns with the shared OpenCLIP 7-prompt ensemble",
            "construct the SIC semantic space with image-center filtering and uniqueness filtering",
            "mine train-split KNN neighbors and train the SIC cluster head with consistency, entropy, and image-semantic losses",
            "run the trained SIC cluster head on the evaluation split",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_sic_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
