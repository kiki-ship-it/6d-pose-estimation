"""PnP 求解器封装（OpenCV）。"""
import cv2
import numpy as np
from typing import Tuple, Optional


def solve_pnp(object_points: np.ndarray,
              image_points: np.ndarray,
              camera_matrix: np.ndarray,
              dist_coeffs: Optional[np.ndarray] = None,
              method: str = 'ITERATIVE',
              use_ransac: bool = True,
              ransac_reproj_threshold: float = 8.0,
              ransac_iterations: int = 100) -> Tuple:
    """Solve for pose using solvePnP / solvePnPRansac.
    Args:
        object_points: (N,3) np.float32 array of 3D model points
        image_points: (N,2) np.float32 array of corresponding 2D image points
        camera_matrix: (3,3) camera intrinsics
        dist_coeffs: distortion coefficients or None
        method: 'ITERATIVE', 'EPNP', 'P3P'...
        use_ransac: if True uses solvePnPRansac
    Returns:
        (success, rvec, tvec, inliers) if use_ransac else (success, rvec, tvec, None)
    """
    flags_map = {
        'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
        'EPNP': cv2.SOLVEPNP_EPNP,
        'P3P': cv2.SOLVEPNP_P3P,
        'AP3P': cv2.SOLVEPNP_AP3P
    }
    flag = flags_map.get(method.upper(), cv2.SOLVEPNP_ITERATIVE)

    object_points = object_points.astype('float32')
    image_points = image_points.astype('float32')

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, camera_matrix, dist_coeffs,
            iterationsCount=ransac_iterations,
            reprojectionError=ransac_reproj_threshold,
            flags=flag
        )
        return success, rvec, tvec, inliers
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs, flags=flag
        )
        return success, rvec, tvec, None
