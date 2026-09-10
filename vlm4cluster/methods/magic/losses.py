from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def entropy_loss(probabilities: torch.Tensor) -> torch.Tensor:
    mean_probability = torch.clamp(probabilities.mean(dim=0), min=1e-9)
    return -(mean_probability * torch.log(mean_probability)).sum()


def cluster_column_cross_entropy(
    image_probabilities: torch.Tensor,
    text_probabilities: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    image_columns = image_probabilities.t()
    text_columns = text_probabilities.t()
    image_columns = image_columns / torch.clamp(image_columns.sum(dim=1, keepdim=True), min=eps)
    text_columns = text_columns / torch.clamp(text_columns.sum(dim=1, keepdim=True), min=eps)
    return -(image_columns * torch.log(torch.clamp(text_columns, min=eps))).sum(dim=1).mean()


class MAGICDistillLoss(nn.Module):
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
        left_columns = left.t()
        right_columns = right.t()
        size = 2 * self.class_num
        joined = torch.cat((left_columns, right_columns), dim=0)
        joined = F.normalize(joined, dim=1)

        similarity = joined @ joined.t() / self.temperature
        positive_left_right = torch.diag(similarity, self.class_num)
        positive_right_left = torch.diag(similarity, -self.class_num)
        positive = torch.cat((positive_left_right, positive_right_left), dim=0).reshape(size, 1)
        negative = similarity[self.mask].reshape(size, -1)

        labels = torch.zeros(size, device=joined.device, dtype=torch.long)
        logits = torch.cat((positive, negative), dim=1)
        return self.criterion(logits, labels) / size
