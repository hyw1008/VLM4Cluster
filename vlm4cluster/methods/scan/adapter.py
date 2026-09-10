from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.scan.core import run_scan_pipeline


@register_method
class SCANMethod(ClusteringMethod):
    name = "scan"
    description = "SCAN (ECCV 2020) with OpenCLIP backbone initialization and train/test-safe selection."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        source_dataset = str(params.get("source_dataset_name", "source dataset"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        eval_step = (
            f"evaluate the selected SCAN head on '{target_dataset}' target-test images"
            if target_dataset is not None
            else "evaluate the selected SCAN head on the test split"
        )
        steps = [
            f"load frozen OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            f"load or encode common frozen OpenCLIP features for '{source_dataset}' source-train images",
            "mine nearest neighbors on the frozen source-train features",
            "train only the SCAN clustering head module with neighbor consistency and entropy regularization",
            "select the best head/checkpoint state using official validation-side SCAN loss",
            "load or encode common frozen OpenCLIP features for the evaluation split",
            eval_step,
        ]
        return steps

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_scan_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
