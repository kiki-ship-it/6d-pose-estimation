"""Keypoint detection module."""
import torch
import torch.nn as nn
from typing import Tuple


class KeypointDetector(nn.Module):
    """CNN-based keypoint detector."""

    def __init__(
        self,
        input_channels: int = 256,
        num_keypoints: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """Initialize keypoint detector.
        
        Args:
            input_channels: Number of input channels
            num_keypoints: Number of keypoints to detect
            hidden_dim: Hidden dimension
            num_layers: Number of convolutional layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Backbone: series of conv layers
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout)
            ])
            in_channels = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Keypoint head: output heatmaps for each keypoint
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_keypoints, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of:
                - heatmaps: (B, K, H, W) - keypoint heatmaps
                - coordinates: (B, K, 2) - keypoint coordinates
        """
        # Extract features
        features = self.backbone(x)
        
        # Generate heatmaps
        heatmaps = self.keypoint_head(features)  # (B, K, H, W)
        
        # Extract keypoint coordinates from heatmaps
        coordinates = self._extract_coordinates(heatmaps)
        
        return heatmaps, coordinates

    @staticmethod
    def _extract_coordinates(heatmaps: torch.Tensor) -> torch.Tensor:
        """Extract coordinates from heatmaps using soft-argmax.
        
        Args:
            heatmaps: Heatmaps (B, K, H, W)
            
        Returns:
            Coordinates (B, K, 2) in normalized format [0, 1]
        """
        batch_size, num_kpts, height, width = heatmaps.shape
        
        # Apply softmax to get probability maps
        heatmaps_flat = heatmaps.view(batch_size, num_kpts, -1)  # (B, K, H*W)
        prob = torch.softmax(heatmaps_flat, dim=2)
        prob = prob.view(batch_size, num_kpts, height, width)
        
        # Create coordinate grids
        y_coords = torch.arange(height, dtype=torch.float32, device=heatmaps.device)
        x_coords = torch.arange(width, dtype=torch.float32, device=heatmaps.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute expected position (soft-argmax)
        coordinates = torch.zeros(batch_size, num_kpts, 2, device=heatmaps.device)
        for b in range(batch_size):
            for k in range(num_kpts):
                coordinates[b, k, 0] = (prob[b, k] * xx).sum() / prob[b, k].sum().clamp(min=1e-8)
                coordinates[b, k, 1] = (prob[b, k] * yy).sum() / prob[b, k].sum().clamp(min=1e-8)
        
        # Normalize to [0, 1]
        coordinates[:, :, 0] /= width
        coordinates[:, :, 1] /= height
        
        return coordinates
