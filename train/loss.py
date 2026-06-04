"""Loss functions for 6D pose estimation."""
import torch
import torch.nn as nn


class KeypointLoss(nn.Module):
    """L2 loss for keypoint coordinates (normalized)."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L2 loss.
        
        Args:
            pred: (B, K, 2) normalized coordinates
            target: (B, K, 2) normalized coordinates
        """
        loss = (pred - target).pow(2).sum(dim=-1)  # (B, K)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
