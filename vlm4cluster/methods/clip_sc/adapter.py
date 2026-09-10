from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.clip_sc.core import _resolve_affinity_kernel, run_clip_sc_pipeline
from vlm4cluster.methods.registry import register_method


@register_method
class CLIPSpectralClusteringMethod(ClusteringMethod):
    name = "clip_sc"
    description = "Spectral clustering on the zero-diagonal dot-product graph of normalized OpenCLIP image features."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    @classmethod
    def resolve_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(params)
        resolved["affinity_kernel"] = _resolve_affinity_kernel(params)
        return resolved

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
            (
                "construct the full normalized feature Gram matrix and set its diagonal to zero"
                if _resolve_affinity_kernel(params) == "dot_product"
                else "construct a weighted mutual top-q RBF graph from the evaluation-split OpenCLIP image features"
            ),
            "run sklearn spectral clustering with the precomputed affinity graph",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_clip_sc_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
