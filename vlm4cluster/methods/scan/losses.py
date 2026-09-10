from __future__ import annotations

import math

from vlm4cluster.utils.deps import require_module


EPS = 1e-8


class MaskedCrossEntropyLoss(require_module("torch.nn", "pip install torch").Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor, target, mask, weight=None, reduction: str = "mean"):
        torch = require_module("torch", "pip install torch")
        if not (mask != 0).any():
            raise ValueError("Mask in MaskedCrossEntropyLoss is all zeros.")
        target = torch.masked_select(target, mask)
        batch_size, num_classes = input_tensor.size()
        selected = target.size(0)
        input_tensor = torch.masked_select(input_tensor, mask.view(batch_size, 1)).view(selected, num_classes)
        return torch.nn.functional.cross_entropy(input_tensor, target, weight=weight, reduction=reduction)


class ConfidenceBasedCE(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, threshold: float, apply_class_balancing: bool) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        torch = require_module("torch", "pip install torch")
        weak_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_prob, dim=1)
        mask = max_prob > self.threshold
        _, num_classes = weak_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        selected = target_masked.size(0)

        if self.apply_class_balancing and selected > 0:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1 / (counts.float() / selected)
            weight = torch.ones(num_classes, device=anchors_strong.device)
            weight[idx] = freq
        else:
            weight = None

        return self.loss(anchors_strong, target, mask, weight=weight, reduction="mean")


def entropy(x, input_as_probabilities: bool):
    torch = require_module("torch", "pip install torch")
    if input_as_probabilities:
        values = torch.clamp(x, min=EPS)
        base = values * torch.log(values)
    else:
        base = torch.nn.functional.softmax(x, dim=1) * torch.nn.functional.log_softmax(x, dim=1)

    if len(base.size()) == 2:
        return -base.sum(dim=1).mean()
    if len(base.size()) == 1:
        return -base.sum()
    raise ValueError(f"Input tensor is {len(base.size())}-Dimensional")


class SCANLoss(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, entropy_weight: float = 2.0) -> None:
        torch = require_module("torch", "pip install torch")
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = torch.nn.BCELoss()
        self.entropy_weight = entropy_weight

    def forward(self, anchors, neighbors):
        torch = require_module("torch", "pip install torch")
        batch_size, num_classes = anchors.size()
        anchors_prob = self.softmax(anchors)
        neighbors_prob = self.softmax(neighbors)
        similarity = torch.bmm(
            anchors_prob.view(batch_size, 1, num_classes),
            neighbors_prob.view(batch_size, num_classes, 1),
        ).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        return total_loss, consistency_loss, entropy_loss


class SimCLRLoss(require_module("torch.nn", "pip install torch").Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        torch = require_module("torch", "pip install torch")
        batch_size, num_views, _ = features.size()
        if num_views != 2:
            raise ValueError("SimCLRLoss expects exactly two views.")

        mask = torch.eye(batch_size, dtype=torch.float32, device=features.device)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]
        dot_product = torch.matmul(anchor, contrast_features.t()) / self.temperature
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=features.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        return -((mask * log_prob).sum(1) / mask.sum(1)).mean()
