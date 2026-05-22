"""Data augmentation transforms."""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import random


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, **kwargs) -> Dict:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs


class Resize:
    """Resize image and scale coordinates."""

    def __init__(self, size: Tuple[int, int]):
        """Initialize.
        
        Args:
            size: Target size (height, width)
        """
        self.size = size

    def __call__(self, image: np.ndarray, bboxes: List, keypoints: List, **kwargs) -> Dict:
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.size[1], self.size[0]))

        scale_x = self.size[1] / orig_w
        scale_y = self.size[0] / orig_h

        bboxes = [[x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y] for x1, y1, x2, y2 in bboxes]
        keypoints = [
            np.array([[u*scale_x, v*scale_y] for u, v in kpts])
            for kpts in keypoints
        ]

        return {'image': image, 'bboxes': bboxes, 'keypoints': keypoints, **kwargs}


class RandomFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, bboxes: List, keypoints: List, **kwargs) -> Dict:
        if random.random() < self.p:
            image = cv2.flip(image, 1)  # Horizontal flip
            h, w = image.shape[:2]

            bboxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in bboxes]
            keypoints = [
                np.array([[w - u, v] for u, v in kpts])
                for kpts in keypoints
            ]

        return {'image': image, 'bboxes': bboxes, 'keypoints': keypoints, **kwargs}


class RandomBrightness:
    """Random brightness adjustment."""

    def __init__(self, factor: float = 0.2):
        self.factor = factor

    def __call__(self, image: np.ndarray, **kwargs) -> Dict:
        if random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-self.factor, self.factor)
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
            image = np.clip(image, 0, 255).astype(np.uint8)

        return {'image': image, **kwargs}


class RandomContrast:
    """Random contrast adjustment."""

    def __init__(self, factor: float = 0.2):
        self.factor = factor

    def __call__(self, image: np.ndarray, **kwargs) -> Dict:
        if random.random() < 0.5:
            contrast_factor = 1.0 + random.uniform(-self.factor, self.factor)
            image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
            image = np.clip(image, 0, 255).astype(np.uint8)

        return {'image': image, **kwargs}


def get_transforms(config: Dict, split: str = 'train') -> Compose:
    """Get data transforms based on config.
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        
    Returns:
        Compose object with transforms
    """
    transforms = []

    if split == 'train':
        aug_config = config.get('augmentation_config', {})
        
        if aug_config.get('random_flip', False):
            transforms.append(RandomFlip(p=0.5))
        
        if aug_config.get('random_brightness', False):
            transforms.append(RandomBrightness(factor=aug_config.get('random_brightness', 0.2)))
        
        if aug_config.get('random_contrast', False):
            transforms.append(RandomContrast(factor=aug_config.get('random_contrast', 0.2)))

    # Always resize
    transforms.append(Resize(size=(config.get('input_size', 640), config.get('input_size', 640))))

    return Compose(transforms)
