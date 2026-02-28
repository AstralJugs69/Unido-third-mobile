from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_var_counts = nn.Parameter(torch.zeros(1))
        self.log_var_measures = nn.Parameter(torch.zeros(1))
        self.log_var_consistency = nn.Parameter(torch.zeros(1))

    def forward(self, loss_counts: torch.Tensor, loss_measures: torch.Tensor, loss_consistency: torch.Tensor) -> torch.Tensor:
        loss = (
            torch.exp(-self.log_var_counts) * loss_counts + self.log_var_counts
            + torch.exp(-self.log_var_measures) * loss_measures + self.log_var_measures
            + torch.exp(-self.log_var_consistency) * loss_consistency + self.log_var_consistency
        )
        return loss


def consistency_loss(pred_counts: torch.Tensor) -> torch.Tensor:
    # Total (0) should be close to Broken + Long + Medium (1..3)
    return torch.nn.functional.l1_loss(pred_counts[:, 1:4].sum(dim=1), pred_counts[:, 0])


def weighted_count_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target)
    huber_like = torch.where(diff < 10.0, 0.5 * diff**2, 10.0 * (diff - 5.0))
    return (huber_like * weights).mean()


def gradnorm_placeholder(*args, **kwargs):
    raise NotImplementedError("GradNorm is intentionally not implemented in this core scope.")
