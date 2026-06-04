"""基础模型正向传播测试（非常简化）。"""
import torch
from models.pose_estimator import PoseEstimator


def test_model_forward():
    model = PoseEstimator()
    x = torch.randn(1,3,640,640)
    out = model(x)
    assert 'keypoints' in out
