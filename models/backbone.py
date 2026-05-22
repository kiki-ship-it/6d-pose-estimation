"""Backbone networks."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction."""

    def __init__(self, depth: int = 50, pretrained: bool = True, freeze: bool = False):
        """Initialize ResNet backbone.
        
        Args:
            depth: ResNet depth (18, 34, 50, 101, 152)
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze backbone parameters
        """
        super().__init__()
        
        if depth == 18:
            self.backbone = models.resnet18(pretrained=pretrained)
        elif depth == 34:
            self.backbone = models.resnet34(pretrained=pretrained)
        elif depth == 50:
            self.backbone = models.resnet50(pretrained=pretrained)
        elif depth == 101:
            self.backbone = models.resnet101(pretrained=pretrained)
        elif depth == 152:
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimension
        self.feat_dim = 2048 if depth >= 50 else 512
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Feature tensor (B, feat_dim, H/32, W/32)
        """
        return self.backbone(x)
