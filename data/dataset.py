"""6D Pose Estimation Dataset."""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset
import json
import os


class PoseDataset(Dataset):
    """Dataset for 6D pose estimation.
    
    Expected directory structure:
    ├── images/
    │   ├── 000000.jpg
    │   └── ...
    ├── masks/
    │   ├── 000000.png
    │   └── ...
    └── annotations.json
    
    annotations.json format:
    {
        "000000": {
            "image_id": "000000",
            "objects": [
                {
                    "category_id": 1,
                    "bbox": [x1, y1, x2, y2],
                    "keypoints": [[u1, v1], [u2, v2], ...],
                    "pose": {
                        "rotation": [[...], [...], [...]],
                        "translation": [x, y, z]
                    }
                }
            ]
        }
    }
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transforms: Optional[object] = None,
        num_keypoints: int = 8,
        image_size: Tuple[int, int] = (640, 640),
    ):
        """Initialize dataset.
        
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            transforms: Image transforms
            num_keypoints: Number of keypoints per object
            image_size: Target image size (height, width)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.num_keypoints = num_keypoints
        self.image_size = image_size

        # Load annotations
        self.annotations = self._load_annotations()
        self.image_ids = sorted(self.annotations.keys())

    def _load_annotations(self) -> Dict:
        """Load annotations from JSON file."""
        ann_file = self.data_root / f'annotations_{self.split}.json'
        
        if not ann_file.exists():
            # If split-specific annotation doesn't exist, use general annotations
            ann_file = self.data_root / 'annotations.json'
            if not ann_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing:
                - 'image': RGB image tensor (3, H, W)
                - 'bboxes': Bounding boxes (N, 4) in format [x1, y1, x2, y2]
                - 'class_ids': Object class IDs (N,)
                - 'keypoints': 2D keypoints (N, K, 2)
                - 'poses': 6D poses containing R (3, 3) and t (3,)
                - 'image_id': Image ID string
        """
        image_id = self.image_ids[idx]
        ann_data = self.annotations[image_id]

        # Load image
        image_path = self.data_root / 'images' / f'{image_id}.jpg'
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Parse objects
        objects = ann_data.get('objects', [])
        
        bboxes = []
        class_ids = []
        keypoints = []
        poses = []

        for obj in objects:
            # Bounding box
            bbox = obj['bbox']  # [x1, y1, x2, y2]
            bboxes.append(bbox)

            # Class ID
            class_ids.append(obj['category_id'])

            # 2D Keypoints
            kpts = np.array(obj['keypoints'], dtype=np.float32)  # (K, 2)
            keypoints.append(kpts)

            # 6D Pose
            pose_data = obj['pose']
            R = np.array(pose_data['rotation'], dtype=np.float32)  # (3, 3)
            t = np.array(pose_data['translation'], dtype=np.float32)  # (3,)
            poses.append({'R': R, 't': t})

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                keypoints=keypoints
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            keypoints = transformed['keypoints']
        else:
            # Default: Resize to target size
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            scale_x = self.image_size[1] / orig_w
            scale_y = self.image_size[0] / orig_h
            
            bboxes = [[x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y] for x1, y1, x2, y2 in bboxes]
            keypoints = [[kpts[:, 0]*scale_x, kpts[:, 1]*scale_y] for kpts in keypoints]
            keypoints = [np.stack(kp, axis=1) for kp in keypoints]

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        bboxes_tensor = torch.FloatTensor(bboxes) if bboxes else torch.zeros((0, 4))
        class_ids_tensor = torch.LongTensor(class_ids) if class_ids else torch.zeros((0,), dtype=torch.long)
        
        # Pad keypoints to fixed size
        keypoints_tensor = torch.zeros((len(keypoints), self.num_keypoints, 2))
        for i, kpts in enumerate(keypoints):
            if len(kpts) > 0:
                kpts_padded = np.zeros((self.num_keypoints, 2))
                kpts_padded[:min(len(kpts), self.num_keypoints)] = kpts[:self.num_keypoints]
                keypoints_tensor[i] = torch.from_numpy(kpts_padded).float()

        return {
            'image': image,
            'bboxes': bboxes_tensor,
            'class_ids': class_ids_tensor,
            'keypoints': keypoints_tensor,
            'poses': poses,
            'image_id': image_id,
            'image_size': torch.tensor(self.image_size, dtype=torch.int32),
        }
