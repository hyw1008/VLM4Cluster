from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.tac.core import run_tac_pipeline


@register_method
class TACMethod(ClusteringMethod):
    name = "tac"
    description = "Text-Aided Clustering from ICML 2024 with WordNet guidance and OpenCLIP backbones."
    category = "image_text"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        variant = "train cluster heads" if bool(params.get("train_cluster_heads", True)) else "training-free concat kmeans"
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
            image_step,
            "load the shared WordNetNouns.csv resource from the data directory and encode its noun pool with the shared OpenCLIP text encoder",
            "select discriminative nouns from train-image semantic centers and retrieve text counterparts",
            f"execute TAC variant: {variant}",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_tac_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
