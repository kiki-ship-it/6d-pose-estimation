"""6D Pose Estimator."""
import torch
import torch.nn as nn
from .backbone import ResNetBackbone
from .detector import YOLODetector
from .keypoint_detector import KeypointDetector
from typing import Dict, List, Tuple


class PoseEstimator(nn.Module):
    """Complete 6D pose estimation model.
    
    Pipeline:
        1. Object Detection (YOLOv8) -> bboxes
        2. Feature Extraction (ResNet) -> features
        3. Keypoint Detection (CNN) -> 2D keypoints
        4. PnP Solving (OpenCV) -> 6D poses (R, t)
    """

    def __init__(
        self,
        backbone_depth: int = 50,
        num_keypoints: int = 8,
        num_classes: int = 21,
        hidden_dim: int = 256,
        freeze_backbone: bool = False,
        pretrained: bool = True
    ):
        """Initialize pose estimator.
        
        Args:
            backbone_depth: ResNet depth (18, 34, 50, 101, 152)
            num_keypoints: Number of keypoints
            num_classes: Number of object classes
            hidden_dim: Hidden dimension
            freeze_backbone: Whether to freeze backbone
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.backbone_depth = backbone_depth
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Backbone for feature extraction
        self.backbone = ResNetBackbone(
            depth=backbone_depth,
            pretrained=pretrained,
            freeze=freeze_backbone
        )
        
        # Keypoint detector
        self.keypoint_detector = KeypointDetector(
            input_channels=self.backbone.feat_dim,
            num_keypoints=num_keypoints,
            hidden_dim=hidden_dim
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Dictionary with:
                - 'features': Feature maps (B, C, H/32, W/32)
                - 'heatmaps': Keypoint heatmaps (B, K, H/32, W/32)
                - 'keypoints': Keypoint coordinates (B, K, 2) normalized
        """
        # Extract features
        features = self.backbone(x)  # (B, 2048, H/32, W/32)
        
        # Detect keypoints
        heatmaps, keypoints = self.keypoint_detector(features)  # (B, K, H, W), (B, K, 2)
        
        return {
            'features': features,
            'heatmaps': heatmaps,
            'keypoints': keypoints
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without keypoint detection.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Feature tensor (B, feat_dim)
        """
        features = self.backbone(x)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
