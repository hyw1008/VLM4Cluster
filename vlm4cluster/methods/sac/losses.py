from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def entropy_loss(probabilities: torch.Tensor) -> torch.Tensor:
    mean_prob = probabilities.mean(dim=0)
    mean_prob = torch.clamp(mean_prob, min=1e-9)
    return -(mean_prob * torch.log(mean_prob)).sum()


def instance_consistency_loss(text_probabilities: torch.Tensor, image_probabilities: torch.Tensor) -> torch.Tensor:
    batch_size, cluster_num = text_probabilities.size()
    similarity = torch.bmm(
        text_probabilities.view(batch_size, 1, cluster_num),
        image_probabilities.view(batch_size, cluster_num, 1),
    ).view(-1)
    return F.binary_cross_entropy(similarity, torch.ones_like(similarity))


def cluster_consistency_loss(text_probabilities: torch.Tensor, image_probabilities: torch.Tensor) -> torch.Tensor:
    text_columns = text_probabilities.t()
    image_columns = image_probabilities.t()
    similarity = torch.sum(text_columns * image_columns, dim=1)
    return F.binary_cross_entropy_with_logits(similarity, torch.ones_like(similarity))


def bidirectional_alignment_loss(
    text_probabilities: torch.Tensor,
    image_probabilities: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return beta * instance_consistency_loss(text_probabilities, image_probabilities) + (
        1.0 - beta
    ) * cluster_consistency_loss(text_probabilities, image_probabilities)


class SACDataContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.1, weight_scale: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight_scale = weight_scale
        self.eps = eps

    def _reliability_weight(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        batch_size = left.size(0)
        features = torch.cat([left, right], dim=0)
        similarity = torch.mm(features, features.t()) / self.temperature
        positives = torch.diag(similarity, batch_size)
        positives = torch.cat([positives, positives], dim=0)
        return torch.sigmoid(self.weight_scale * positives)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_weight_view: torch.Tensor,
        right_weight_view: torch.Tensor,
        left_weight_anchor: torch.Tensor,
        right_weight_anchor: torch.Tensor,
        *,
        weighted: bool,
        alpha: float,
    ) -> torch.Tensor:
        batch_size = left.size(0)
        left = F.normalize(left, p=1, dim=1)
        right = F.normalize(right, p=1, dim=1)
        left_weight_view = F.normalize(left_weight_view, p=1, dim=1)
        right_weight_view = F.normalize(right_weight_view, p=1, dim=1)
        left_weight_anchor = F.normalize(left_weight_anchor, p=1, dim=1)
        right_weight_anchor = F.normalize(right_weight_anchor, p=1, dim=1)

        features = torch.cat([left, right], dim=0)
        similarity = torch.mm(features, features.t()) / self.temperature
        positive_indices = torch.arange(batch_size, device=left.device)
        positive_indices = torch.cat([positive_indices + batch_size, positive_indices], dim=0)
        log_probabilities = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        sample_losses = -log_probabilities[torch.arange(2 * batch_size, device=left.device), positive_indices]

        if not weighted:
            return sample_losses.mean()

        alpha_tensor = torch.as_tensor(alpha, device=left.device, dtype=left.dtype)
        base_anchor = self._reliability_weight(left_weight_anchor, right_weight_anchor)
        base_view = self._reliability_weight(left_weight_view, right_weight_view)
        weights = alpha_tensor * base_anchor + (1.0 - alpha_tensor) * base_view
        return torch.sum(weights * sample_losses) / torch.clamp(torch.sum(weights), min=self.eps)
