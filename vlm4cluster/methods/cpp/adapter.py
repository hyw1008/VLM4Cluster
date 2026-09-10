from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.cpp.core import _resolve_head_mode, run_cpp_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class CPPMethod(ClusteringMethod):
    name = "cpp"
    description = "CPP/MLC rate-reduction clustering on shared OpenCLIP image features."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        source_dataset = str(params.get("source_dataset_name", "source dataset"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        head_mode = _resolve_head_mode(params)
        training_step = (
            "train CPP in single-head mode; membership Pi is built directly from subspace Z"
            if head_mode == "single_head"
            else "train CPP with the two-head MLC objective; membership Pi is built from the cluster/logit head"
        )
        evaluation_step = (
            f"after each epoch, run spectral clustering on '{target_dataset}' target-test membership matrix and keep the best test epoch"
            if target_dataset is not None
            else "after each epoch, run spectral clustering on the test split membership matrix and keep the best test epoch"
        )
        return [
            f"load OpenCLIP visual backbone '{backbone}' pretrained on '{pretraining}'",
            f"embed '{source_dataset}' source-train images with the shared raw OpenCLIP feature cache",
            training_step,
            evaluation_step,
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_cpp_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
