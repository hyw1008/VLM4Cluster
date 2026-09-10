from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.ntk_sc.core import run_ntk_sc_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class NTKSCMethod(ClusteringMethod):
    name = "ntk_sc"
    description = "Spectral clustering with proxy NTK and RED over OpenCLIP vision-language representations."
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
            "load the shared WordNetNouns.csv resource from the data directory and encode all nouns with the shared OpenCLIP text encoder",
            image_step,
            "filter discriminative nouns from train-image proxy clusters",
            "run GPU-first RED by default: dense GPU RED on small eval splits, cached prompt-wise dense pNTK plus GPU out-of-core RED on large eval splits, with sklearn spectral clustering at the end",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_ntk_sc_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
