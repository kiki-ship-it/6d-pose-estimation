"""测试 dataset.py 的基础单元测试（非常简化）。"""
import tempfile
import numpy as np
from pathlib import Path
from data.dataset import PoseDataset


def test_dataset_load_minimal():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        images = root / 'images'
        images.mkdir()
        img_path = images / '000000.jpg'
        import cv2
        import numpy as np
        cv2.imwrite(str(img_path), np.zeros((100,100,3), dtype='uint8'))

        ann = {
            '000000': {
                'image_id': '000000',
                'objects': []
            }
        }
        import json
        with open(root / 'annotations.json', 'w') as f:
            json.dump(ann, f)

        ds = PoseDataset(str(root))
        assert len(ds) == 1
