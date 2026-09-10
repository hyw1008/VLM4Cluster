from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.temi.core import run_temi_pipeline


@register_method
class TEMIMethod(ClusteringMethod):
    name = "temi"
    description = "TEMI clustering head training on shared OpenCLIP embeddings."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        source_dataset = str(params.get("source_dataset_name", "source dataset"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        evaluation_step = (
            f"evaluate the EMA teacher head on '{target_dataset}' target-test embeddings"
            if target_dataset is not None
            else "evaluate the EMA teacher head on the test split embeddings"
        )
        return [
            f"load OpenCLIP visual backbone '{backbone}' pretrained on '{pretraining}'",
            f"embed '{source_dataset}' source-train images with the shared raw OpenCLIP feature cache",
            "build a TEMI train KNN graph from source-train embeddings",
            "train the TEMI student/teacher multi-head clustering head on source-train KNN pairs",
            evaluation_step,
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_temi_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
