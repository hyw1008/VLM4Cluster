from __future__ import annotations

from dataclasses import dataclass

from vlm4cluster.utils.deps import require_module


@dataclass(frozen=True, slots=True)
class OpenCLIPSpec:
    benchmark_pretraining: str
    benchmark_backbone: str
    model_name: str
    pretrained: str
    tokenizer_model_name: str

    @property
    def cache_key(self) -> str:
        pretraining = self.benchmark_pretraining.lower()
        backbone = self.benchmark_backbone.lower().replace("/", "-")
        return f"{pretraining}__{backbone}"


@dataclass(slots=True)
class OpenCLIPBundle:
    spec: OpenCLIPSpec
    model: object
    preprocess: object
    tokenizer: object


_MODEL_POOL: dict[tuple[str, str], OpenCLIPSpec] = {
    ("LAION400M", "ViT-B/32"): OpenCLIPSpec(
        benchmark_pretraining="LAION400M",
        benchmark_backbone="ViT-B/32",
        model_name="ViT-B-32-quickgelu",
        pretrained="laion400m_e32",
        tokenizer_model_name="ViT-B-32",
    ),
    ("LAION2B", "ViT-B/32"): OpenCLIPSpec(
        benchmark_pretraining="LAION2B",
        benchmark_backbone="ViT-B/32",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        tokenizer_model_name="ViT-B-32",
    ),
    ("LAION400M", "ViT-B/16"): OpenCLIPSpec(
        benchmark_pretraining="LAION400M",
        benchmark_backbone="ViT-B/16",
        model_name="ViT-B-16",
        pretrained="laion400m_e32",
        tokenizer_model_name="ViT-B-16",
    ),
    ("LAION2B", "ViT-B/16"): OpenCLIPSpec(
        benchmark_pretraining="LAION2B",
        benchmark_backbone="ViT-B/16",
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        tokenizer_model_name="ViT-B-16",
    ),
    ("SigLIP", "ViT-B/16"): OpenCLIPSpec(
        benchmark_pretraining="SigLIP",
        benchmark_backbone="ViT-B/16",
        model_name="ViT-B-16-SigLIP",
        pretrained="webli",
        tokenizer_model_name="ViT-B-16-SigLIP",
    ),
    ("LAION400M", "ViT-L/14"): OpenCLIPSpec(
        benchmark_pretraining="LAION400M",
        benchmark_backbone="ViT-L/14",
        model_name="ViT-L-14",
        pretrained="laion400m_e32",
        tokenizer_model_name="ViT-L-14",
    ),
    ("LAION2B", "ViT-L/14"): OpenCLIPSpec(
        benchmark_pretraining="LAION2B",
        benchmark_backbone="ViT-L/14",
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        tokenizer_model_name="ViT-L-14",
    ),
}


def _normalize_pretraining(name: str) -> str:
    key = name.strip().upper().replace("-", "")
    mapping = {
        "LAION400M": "LAION400M",
        "LAION2B": "LAION2B",
        "SIGLIP": "SigLIP",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported OpenCLIP pretraining '{name}'.") from exc


def _normalize_backbone(name: str) -> str:
    key = name.strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "VITB/32": "ViT-B/32",
        "VITB32": "ViT-B/32",
        "VITB/16": "ViT-B/16",
        "VITB16": "ViT-B/16",
        "VITL/14": "ViT-L/14",
        "VITL14": "ViT-L/14",
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported OpenCLIP backbone '{name}'.") from exc


def get_openclip_spec(pretraining: str, backbone: str) -> OpenCLIPSpec:
    key = (_normalize_pretraining(pretraining), _normalize_backbone(backbone))
    try:
        return _MODEL_POOL[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported OpenCLIP combination pretraining={pretraining!r}, backbone={backbone!r}."
        ) from exc


def load_openclip_bundle(pretraining: str, backbone: str, device: str) -> OpenCLIPBundle:
    open_clip = require_module("open_clip", "pip install open_clip_torch")
    spec = get_openclip_spec(pretraining, backbone)
    model, _, preprocess = open_clip.create_model_and_transforms(spec.model_name, pretrained=spec.pretrained)
    tokenizer = open_clip.get_tokenizer(spec.tokenizer_model_name)

    torch = require_module("torch", "pip install torch")
    model.eval()
    model.to(torch.device(device))
    return OpenCLIPBundle(spec=spec, model=model, preprocess=preprocess, tokenizer=tokenizer)
