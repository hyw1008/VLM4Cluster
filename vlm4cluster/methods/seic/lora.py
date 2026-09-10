from __future__ import annotations

import math
from dataclasses import dataclass

from vlm4cluster.utils.deps import require_module


@dataclass(slots=True)
class LoRAInjectionResult:
    modules_replaced: int
    trainable_parameters: int


class LoRALinearAdapter(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, in_features: int, out_features: int, rank: int, *, alpha: float, dropout: float) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = torch.nn.Dropout(float(dropout)) if dropout > 0 else torch.nn.Identity()
        self.down = torch.nn.Linear(in_features, self.rank, bias=False)
        self.up = torch.nn.Linear(self.rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(self.dropout(x))) * self.scaling


class LoRAMultiheadAttention(require_module("torch.nn", "pip install torch").Module):
    """A focused LoRA wrapper for OpenCLIP ViT self-attention blocks."""

    def __init__(self, base_attention, *, rank: int, alpha: float, dropout: float) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        if getattr(base_attention, "bias_k", None) is not None or getattr(base_attention, "bias_v", None) is not None:
            raise ValueError("SEIC LoRA wrapper does not support MultiheadAttention bias_k/bias_v.")
        if bool(getattr(base_attention, "add_zero_attn", False)):
            raise ValueError("SEIC LoRA wrapper does not support add_zero_attn=True.")
        if int(base_attention.embed_dim) % int(base_attention.num_heads) != 0:
            raise ValueError("MultiheadAttention embed_dim must be divisible by num_heads.")

        self.base_attention = base_attention
        for parameter in self.base_attention.parameters():
            parameter.requires_grad = False

        self.embed_dim = int(base_attention.embed_dim)
        self.num_heads = int(base_attention.num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = bool(getattr(base_attention, "batch_first", False))
        self.dropout = float(getattr(base_attention, "dropout", 0.0))
        self.lora_q = LoRALinearAdapter(self.embed_dim, self.embed_dim, rank, alpha=alpha, dropout=dropout)
        self.lora_v = LoRALinearAdapter(self.embed_dim, self.embed_dim, rank, alpha=alpha, dropout=dropout)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        del is_causal
        torch = require_module("torch", "pip install torch")
        functional = require_module("torch.nn.functional", "pip install torch")

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        target_len, batch_size, embed_dim = query.shape
        source_len = key.shape[0]
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected query embed_dim={self.embed_dim}, got {embed_dim}.")

        q, k, v = self._project_qkv(query, key, value)
        q = q + self.lora_q(query)
        v = v + self.lora_v(value)

        q = q.contiguous().view(target_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(source_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(source_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = _apply_attention_masks(scores, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        weights = torch.softmax(scores, dim=-1)
        weights = functional.dropout(weights, p=self.dropout, training=self.training)
        attention = torch.matmul(weights, v)
        attention = attention.permute(2, 0, 1, 3).contiguous().view(target_len, batch_size, self.embed_dim)
        output = self.base_attention.out_proj(attention)
        if self.batch_first:
            output = output.transpose(0, 1)

        if not need_weights:
            return output, None
        if average_attn_weights:
            returned_weights = weights.mean(dim=1)
        else:
            returned_weights = weights.reshape(batch_size * self.num_heads, target_len, source_len)
        return output, returned_weights

    def _project_qkv(self, query, key, value):
        functional = require_module("torch.nn.functional", "pip install torch")

        if getattr(self.base_attention, "_qkv_same_embed_dim", True):
            weight = self.base_attention.in_proj_weight
            bias = self.base_attention.in_proj_bias
            q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
            if bias is None:
                q_bias = k_bias = v_bias = None
            else:
                q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
        else:
            q_weight = self.base_attention.q_proj_weight
            k_weight = self.base_attention.k_proj_weight
            v_weight = self.base_attention.v_proj_weight
            bias = self.base_attention.in_proj_bias
            if bias is None:
                q_bias = k_bias = v_bias = None
            else:
                q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
        return (
            functional.linear(query, q_weight, q_bias),
            functional.linear(key, k_weight, k_bias),
            functional.linear(value, v_weight, v_bias),
        )


def inject_lora_into_visual_attention(visual_module, *, rank: int, alpha: float, dropout: float) -> LoRAInjectionResult:
    torch = require_module("torch", "pip install torch")

    replacements = []
    for module_name, module in visual_module.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            replacements.append((module_name, module))
    if not replacements:
        raise ValueError(
            "SEIC could not find torch.nn.MultiheadAttention modules under OpenCLIP visual. "
            "This OpenCLIP backbone needs a custom LoRA adapter."
        )

    for parameter in visual_module.parameters():
        parameter.requires_grad = False

    for module_name, module in replacements:
        parent, child_name = _resolve_parent_module(visual_module, module_name)
        setattr(parent, child_name, LoRAMultiheadAttention(module, rank=rank, alpha=alpha, dropout=dropout))

    trainable_parameters = sum(
        parameter.numel() for parameter in visual_module.parameters() if parameter.requires_grad
    )
    return LoRAInjectionResult(modules_replaced=len(replacements), trainable_parameters=int(trainable_parameters))


def _resolve_parent_module(root_module, module_name: str):
    if "." not in module_name:
        return root_module, module_name
    parent_path, child_name = module_name.rsplit(".", 1)
    parent = root_module
    for part in parent_path.split("."):
        parent = getattr(parent, part)
    return parent, child_name


def _apply_attention_masks(scores, *, attn_mask, key_padding_mask):
    torch = require_module("torch", "pip install torch")

    if attn_mask is not None:
        mask = attn_mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                batch_heads, target_len, source_len = mask.shape
                mask = mask.view(scores.size(0), scores.size(1), target_len, source_len)
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            mask = mask.to(dtype=scores.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                batch_heads, target_len, source_len = mask.shape
                mask = mask.view(scores.size(0), scores.size(1), target_len, source_len)
            scores = scores + mask

    if key_padding_mask is not None:
        mask = key_padding_mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
        else:
            scores = scores + mask.to(dtype=scores.dtype)[:, None, None, :]
    return scores
