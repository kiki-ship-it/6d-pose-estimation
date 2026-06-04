"""推理端到端小规模测试（不依赖YOLO权重，跳过检测）"""
import numpy as np
from inference.predictor import Predictor


def test_predictor_smoke():
    pred = Predictor(model_path=None, camera_matrix=np.eye(3), obj_models={})
    # Create dummy image
    img = np.zeros((480,640,3), dtype=np.uint8)
    res = pred.predict(img)
    assert isinstance(res, list)
