from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.pro_dsc.core import run_pro_dsc_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class PRODSCMethod(ClusteringMethod):
    name = "pro_dsc"
    description = "PRO-DSC: a principled deep subspace clustering framework on shared OpenCLIP image features."
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
            f"run final spectral clustering on '{target_dataset}' target-test logits affinity"
            if target_dataset is not None
            else "run final spectral clustering on the test split logits affinity"
        )
        return [
            f"load OpenCLIP visual backbone '{backbone}' pretrained on '{pretraining}'",
            f"embed '{source_dataset}' source-train images with the shared raw OpenCLIP feature cache",
            "train PRO-DSC on the source-train OpenCLIP features with TCR warmup and Sinkhorn-projected self-expression",
            evaluation_step,
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_pro_dsc_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
