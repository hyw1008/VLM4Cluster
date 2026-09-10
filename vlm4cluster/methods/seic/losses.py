from __future__ import annotations

from dataclasses import dataclass

from vlm4cluster.utils.deps import require_module


@dataclass(slots=True)
class AlignmentLossOutput:
    total: object
    instance: object
    assignment: object
    center: object
    balance: object


def bidirectional_contrastive_loss(left, right, temperature):
    torch = require_module("torch", "pip install torch")
    functional = require_module("torch.nn.functional", "pip install torch")

    if left.size(0) < 2:
        return left.new_zeros(())
    left = functional.normalize(left, dim=1)
    right = functional.normalize(right, dim=1)
    logits = left @ right.t() / torch.clamp(torch.as_tensor(temperature, device=left.device), min=1.0e-6)
    labels = torch.arange(left.size(0), device=left.device)
    return (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.t(), labels)) / 2.0


def trainable_temperature_contrastive_loss(left, right, logit_scale):
    torch = require_module("torch", "pip install torch")
    functional = require_module("torch.nn.functional", "pip install torch")

    if left.size(0) < 2:
        return left.new_zeros(())
    left = functional.normalize(left, dim=1)
    right = functional.normalize(right, dim=1)
    scale = torch.clamp(logit_scale.exp(), max=100.0)
    logits = scale * (left @ right.t())
    labels = torch.arange(left.size(0), device=left.device)
    return (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.t(), labels)) / 2.0


def assignment_contrastive_loss(image_probabilities, text_probabilities, temperature: float):
    functional = require_module("torch.nn.functional", "pip install torch")

    image_columns = functional.normalize(image_probabilities.t(), dim=1)
    text_columns = functional.normalize(text_probabilities.t(), dim=1)
    return bidirectional_contrastive_loss(image_columns, text_columns, temperature)


def probability_weighted_centers(projected_features, probabilities):
    torch = require_module("torch", "pip install torch")

    assignments = torch.argmax(probabilities, dim=1)
    num_clusters = probabilities.size(1)
    one_hot = functional_one_hot(assignments, num_classes=num_clusters).to(projected_features.dtype)
    weights = probabilities * one_hot
    centers = weights.t() @ projected_features
    valid = weights.sum(dim=0) > 1.0e-8
    return centers, valid


def center_contrastive_loss(image_projected, text_projected, image_probabilities, text_probabilities, temperature: float):
    image_centers, image_valid = probability_weighted_centers(image_projected, image_probabilities)
    text_centers, text_valid = probability_weighted_centers(text_projected, text_probabilities)
    valid = image_valid & text_valid
    if int(valid.sum().item()) < 2:
        return image_projected.new_zeros(())
    return bidirectional_contrastive_loss(image_centers[valid], text_centers[valid], temperature)


def dynamic_balance_loss(
    image_probabilities,
    text_probabilities,
    assignment_history,
    *,
    history_floor: float,
    max_history_weight: float,
    eps: float = 1.0e-8,
):
    torch = require_module("torch", "pip install torch")

    image_mean = torch.clamp(image_probabilities.mean(dim=0), min=eps)
    text_mean = torch.clamp(text_probabilities.mean(dim=0), min=eps)
    history = torch.clamp(assignment_history.to(image_probabilities.device), min=max(float(history_floor), eps))
    history = history / torch.clamp(history.sum(), min=eps)
    uniform_mass = 1.0 / float(history.numel())
    history_weights = torch.clamp(uniform_mass / history, max=float(max_history_weight))
    history_weights = history_weights / torch.clamp(history_weights.mean(), min=eps)
    image_entropy = -(history_weights * image_mean * torch.log(image_mean)).sum()
    text_entropy = -(history_weights * text_mean * torch.log(text_mean)).sum()
    return -(image_entropy + text_entropy)


def alignment_loss(
    *,
    image_projected,
    text_projected,
    image_probabilities,
    text_probabilities,
    assignment_history,
    instance_logit_scale,
    assignment_temperature: float,
    center_temperature: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    balance_history_floor: float,
    balance_max_weight: float,
) -> AlignmentLossOutput:
    instance = trainable_temperature_contrastive_loss(image_projected, text_projected, instance_logit_scale)
    assignment = assignment_contrastive_loss(image_probabilities, text_probabilities, assignment_temperature)
    center = center_contrastive_loss(
        image_projected,
        text_projected,
        image_probabilities,
        text_probabilities,
        center_temperature,
    )
    balance = dynamic_balance_loss(
        image_probabilities,
        text_probabilities,
        assignment_history,
        history_floor=balance_history_floor,
        max_history_weight=balance_max_weight,
    )
    total = alpha * instance + beta * assignment + gamma * center + delta * balance
    return AlignmentLossOutput(total=total, instance=instance, assignment=assignment, center=center, balance=balance)


def self_enhancement_loss(strong_logits, pseudo_labels, weights):
    functional = require_module("torch.nn.functional", "pip install torch")

    losses = functional.cross_entropy(strong_logits, pseudo_labels, reduction="none")
    return (losses * weights).mean()


def functional_one_hot(labels, *, num_classes: int):
    functional = require_module("torch.nn.functional", "pip install torch")
    return functional.one_hot(labels, num_classes=num_classes)
