from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.ssc_omp.core import run_ssc_omp_pipeline


@register_method
class SSCOMPMethod(ClusteringMethod):
    name = "ssc_omp"
    description = "Sparse Subspace Clustering by Orthogonal Matching Pursuit on OpenCLIP image features."
    category = "image_only"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

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
            "compute the SSC-OMP self-representation matrix on the evaluation split",
            "construct the affinity matrix and run spectral clustering",
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_ssc_omp_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
