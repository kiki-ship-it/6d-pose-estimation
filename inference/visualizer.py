"""Visualization utilities for pose estimation results."""
import cv2
import numpy as np
from typing import Tuple


def draw_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int], label: str = '', color=(0, 255, 0)) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_keypoints(img: np.ndarray, keypoints: np.ndarray, color=(0, 0, 255)) -> None:
    for (x, y) in keypoints.astype(int):
        cv2.circle(img, (int(x), int(y)), 3, color, -1)


def draw_axes(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, length: float = 0.05) -> None:
    """Draw 3D axes at the object pose.
    """
    # Define axis points in 3D (meters)
    axis = np.float32([[length,0,0],[0,length,0],[0,0,length]]).reshape(-1,3)
    origin = np.float32([[0,0,0]]).reshape(-1,3)
    points_3d = np.vstack([origin, axis])

    imgpts, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1,2).astype(int)

    origin_pt = tuple(imgpts[0])
    x_pt = tuple(imgpts[1])
    y_pt = tuple(imgpts[2])
    z_pt = tuple(imgpts[3])

    cv2.line(img, origin_pt, x_pt, (0,0,255), 2) # X - red
    cv2.line(img, origin_pt, y_pt, (0,255,0), 2) # Y - green
    cv2.line(img, origin_pt, z_pt, (255,0,0), 2) # Z - blue


def visualize_results(img: np.ndarray, results: list, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    vis = img.copy()
    for res in results:
        bbox = res.get('bbox')
        cls = res.get('cls', '')
        draw_bbox(vis, bbox, label=str(cls))
        if 'keypoints' in res:
            draw_keypoints(vis, res['keypoints'])
        if 'rvec' in res and 'tvec' in res:
            draw_axes(vis, camera_matrix, dist_coeffs, res['rvec'], res['tvec'])
    return vis
