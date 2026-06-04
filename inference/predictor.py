"""推理器：整合检测器、关键点检测器与PnP，输出6D位姿。"""
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional
from models.detector import YOLODetector
from models.pose_estimator import PoseEstimator
from inference.pnp_solver import solve_pnp


class Predictor:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 detector_name: str = 'yolov8m',
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None,
                 obj_models: Optional[Dict[int, np.ndarray]] = None):
        """初始化推理器。
        Args:
            model_path: 关键点/骨干网络权重（.pth）或 None
            obj_models: dict[class_id] -> (K,3) 模型关键点的3D坐标
        """
        self.device = device
        self.detector = YOLODetector(model_name=detector_name, device=device)
        self.model = PoseEstimator().to(device)
        if model_path:
            ckpt = torch.load(model_path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
        self.model.eval()

        # Camera
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.eye(3, dtype=np.float32)
        self.dist_coeffs = dist_coeffs

        # obj_models: mapping class_id -> (K,3)
        self.obj_models = obj_models or {}

    def predict(self, image: np.ndarray, conf: float = 0.5, iou: float = 0.45) -> List[Dict[str, Any]]:
        """对单张RGB图像做6D姿态推理，返回每个检测的姿态信息。"""
        # 1) 检测
        dets = self.detector.detect(image, conf=conf, iou=iou)

        results = []
        for det in dets:
            x1, y1, x2, y2 = map(int, det['bbox'])
            h = max(y2 - y1, 1)
            w = max(x2 - x1, 1)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 2) 预处理并送入关键点检测
            inp = cv2.resize(crop, (640, 640))
            inp_t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            inp_t = inp_t.to(self.device)
            with torch.no_grad():
                out = self.model(inp_t)
                keypoints_norm = out['keypoints'][0].cpu().numpy()  # (K,2) normalized (x in [0,1], y in [0,1])

            # 转成原图像像素坐标（基于 crop 尺寸）
            keypoints_px = np.zeros_like(keypoints_norm)
            keypoints_px[:, 0] = keypoints_norm[:, 0] * w + x1
            keypoints_px[:, 1] = keypoints_norm[:, 1] * h + y1

            class_id = det['cls']
            obj_model_kpts = self.obj_models.get(class_id, None)
            if obj_model_kpts is None:
                results.append({'bbox': det['bbox'], 'conf': det['conf'], 'cls': class_id, 'keypoints': keypoints_px})
                continue

            # 3) PnP
            success, rvec, tvec, inliers = solve_pnp(
                object_points=obj_model_kpts.astype(np.float32),
                image_points=keypoints_px.astype(np.float32),
                camera_matrix=self.camera_matrix.astype(np.float32),
                dist_coeffs=self.dist_coeffs,
                use_ransac=True
            )

            if success is False:
                continue

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3,)

            results.append({
                'bbox': det['bbox'],
                'conf': det['conf'],
                'cls': class_id,
                'rvec': rvec,
                'tvec': tvec,
                'R': R,
                't': t,
                'keypoints': keypoints_px,
                'inliers': inliers
            })

        return results
