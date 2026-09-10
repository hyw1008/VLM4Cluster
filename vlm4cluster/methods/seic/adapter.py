from __future__ import annotations

from typing import Any

from vlm4cluster.methods.base import ClusteringMethod, MethodInputs, MethodResult
from vlm4cluster.methods.registry import register_method
from vlm4cluster.methods.seic.core import _param_bool, run_seic_pipeline


@register_method
class SEICMethod(ClusteringMethod):
    name = "seic"
    description = "Self-Enhanced Image Clustering with cross-modal semantic consistency and LoRA fine-tuning."
    category = "image_text"
    requires_raw_images = True
    requires_image_features = False
    requires_text_features = False

    def plan_steps(self, params: dict[str, Any]) -> list[str]:
        pretraining = str(params.get("openclip_pretraining", "LAION400M"))
        backbone = str(params.get("openclip_backbone", "ViT-B/32"))
        k1 = int(params.get("k1", params.get("nouns_per_initial_center", 200)))
        k2 = int(params.get("k2", params.get("nouns_per_image", 50)))
        stage1_only = _param_bool(params.get("stage1_only"), False)
        stage2_enabled = (not stage1_only) and _param_bool(params.get("stage2", params.get("self_enhance")), False)
        return [
            f"load OpenCLIP model '{backbone}' pretrained on '{pretraining}'",
            "encode source-train and evaluation images with the frozen OpenCLIP image encoder",
            f"encode WordNet nouns and build SEIC image-text pairs with k1={k1}, k2={k2}",
            "train image/text projection and clustering heads with SEIC cross-modal consistency losses",
            (
                "inject LoRA into OpenCLIP visual attention and self-enhance with confidence-weighted pseudo-labels"
                if stage2_enabled
                else "skip self-enhanced LoRA fine-tuning and return the Stage-1 image-head predictions"
            ),
        ]

    def run(self, inputs: MethodInputs, params: dict[str, Any]) -> MethodResult:
        outputs = run_seic_pipeline(inputs, params)
        return MethodResult(
            method_name=self.name,
            predictions=outputs.predictions,
            evaluation_labels=outputs.evaluation_labels,
            evaluation_split=outputs.evaluation_split,
            metadata=outputs.metadata,
        )
