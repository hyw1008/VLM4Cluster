from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def entropy_loss(probabilities: torch.Tensor) -> torch.Tensor:
    mean_prob = probabilities.mean(dim=0)
    mean_prob = torch.clamp(mean_prob, min=1e-9)
    return -(mean_prob * torch.log(mean_prob)).sum()


def consistency_loss(text_probabilities: torch.Tensor, image_probabilities: torch.Tensor) -> torch.Tensor:
    batch_size, cluster_num = text_probabilities.size()
    similarity = torch.bmm(
        text_probabilities.view(batch_size, 1, cluster_num),
        image_probabilities.view(batch_size, cluster_num, 1),
    ).squeeze()
    return F.binary_cross_entropy(similarity, torch.ones_like(similarity))


class TACDistillLoss(nn.Module):
    def __init__(self, class_num: int, temperature: float) -> None:
        super().__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.register_buffer("mask", self._build_mask(class_num), persistent=False)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _build_mask(self, class_num: int) -> torch.Tensor:
        size = 2 * class_num
        mask = torch.ones((size, size), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for index in range(class_num):
            mask[index, class_num + index] = False
            mask[class_num + index, index] = False
        return mask

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.t()
        right = right.t()
        size = 2 * self.class_num
        joined = torch.cat((left, right), dim=0)
        joined = F.normalize(joined, dim=1)

        similarity = joined @ joined.t() / self.temperature
        positive_left_right = torch.diag(similarity, self.class_num)
        positive_right_left = torch.diag(similarity, -self.class_num)
        positive = torch.cat((positive_left_right, positive_right_left), dim=0).reshape(size, 1)
        negative = similarity[self.mask].reshape(size, -1)

        labels = torch.zeros(size, device=joined.device, dtype=torch.long)
        logits = torch.cat((positive, negative), dim=1)
        return self.criterion(logits, labels) / size
