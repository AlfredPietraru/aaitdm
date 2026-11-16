"""Loss functions for multiclass image classification."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Softmax focal loss for logits [batch, classes] and integer labels.

    gamma=0 and alpha=None recover ordinary cross entropy. Optional alpha
    supplies one non-negative weight per class, in dataset class order.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        weights = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        if weights is not None and (
            weights.ndim != 1 or weights.numel() == 0
            or not torch.isfinite(weights).all() or (weights < 0).any()
        ):
            raise ValueError("alpha must contain finite non-negative class weights.")
        self.register_buffer("alpha", weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("Expected logits [batch, classes] and labels [batch].")
        if self.alpha is not None and self.alpha.numel() != logits.shape[1]:
            raise ValueError("alpha must contain one weight per output class.")
        # Compute p_t before class weighting: alpha must not change the focal factor.
        log_pt = F.log_softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        loss = (-log_pt) * (-torch.expm1(log_pt)).pow(self.gamma)
        if self.alpha is not None:
            loss = loss * self.alpha[labels]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
