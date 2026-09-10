from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.idc.core import run_idc_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class IDCMethod(ClusteringMethod):
    name = "idc"
    description = "Interactive Deep Clustering via Value Mining, adapted as a standalone OpenCLIP-feature method."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        target_dataset = params.get("target_dataset_name", params.get("target_dataset"))
        steps = [
            f"load OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            "embed source train/test images with the shared OpenCLIP image encoder",
            "run K-Means initialisation on the train split and warm up the IDC cluster head",
            "mine valuable train samples with the IDC hardness/representativeness/diversity rule",
            "simulate oracle feedback on the train split and fine-tune the cluster head",
        ]
        if target_dataset is not None:
            steps.append(f"evaluate the fine-tuned cluster head on target dataset '{target_dataset}'")
        else:
            steps.append("evaluate the fine-tuned cluster head on the test split")
        return steps

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_idc_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
