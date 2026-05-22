"""Object detection module using YOLOv8."""
import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import List, Tuple, Dict
import cv2
import numpy as np


class YOLODetector:
    """YOLO-based object detector for 6D pose estimation."""

    def __init__(self, model_name: str = 'yolov8m', num_classes: int = 21, device: str = 'cuda'):
        """Initialize detector.
        
        Args:
            model_name: YOLOv8 model name (nano, small, medium, large, xlarge)
            num_classes: Number of object classes
            device: Device ('cuda' or 'cpu')
        """
        self.model = YOLO(f'{model_name}.pt')
        self.num_classes = num_classes
        self.device = device

    def detect(self, image: np.ndarray, conf: float = 0.5, iou: float = 0.45) -> List[Dict]:
        """Detect objects in image.
        
        Args:
            image: Input image (H, W, 3) in RGB
            conf: Confidence threshold
            iou: NMS IoU threshold
            
        Returns:
            List of detections with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'conf': confidence score
                - 'cls': class ID
        """
        results = self.model(image, conf=conf, iou=iou, device=self.device)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'cls': cls
                })
        
        return detections

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            image: Input tensor (B, 3, H, W)
            
        Returns:
            Tuple of (predictions, features)
        """
        # This is typically used in training mode
        results = self.model(image)
        return results
