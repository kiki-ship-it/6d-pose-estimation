"""Evaluation metrics for pose estimation."""
import numpy as np
from typing import Dict, Any


def add_error(pred_R, pred_t, gt_R, gt_t, model_points):
    """Compute ADD error for non-symmetric objects.
    
    Args:
        pred_R: Predicted rotation (3x3)
        pred_t: Predicted translation (3,)
        gt_R: Ground truth rotation (3x3)
        gt_t: Ground truth translation (3,)
        model_points: (N, 3) 3D model points
    
    Returns:
        ADD error (float)
    """
    pred_pts = (pred_R @ model_points.T).T + pred_t.reshape(1, 3)
    gt_pts = (gt_R @ model_points.T).T + gt_t.reshape(1, 3)
    return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1))
