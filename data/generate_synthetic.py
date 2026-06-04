"""Generate small synthetic dataset for 6D pose estimation.

Generates simple rendered images with a single cuboid object, projects predefined 3D keypoints
and saves annotations in the expected format.
"""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import cv2


def random_rotation(alpha_range=(-30, 30), beta_range=(-30, 30), gamma_range=(-180, 180)):
    # Euler angles in degrees
    a = math.radians(random.uniform(*alpha_range))
    b = math.radians(random.uniform(*beta_range))
    g = math.radians(random.uniform(*gamma_range))

    # Rotation matrices around x,y,z
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def create_camera_matrix(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K


def project_points(points_3d, R, t, K):
    # points_3d: (N,3)
    pts_cam = (R @ points_3d.T).T + t.reshape(1, 3)
    pts_proj = (K @ pts_cam.T).T
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2:3]
    return pts_proj


def render_simple_object(img, pts2d, color=(0, 200, 80)):
    # Draw convex hull of 2D points for simple appearance
    pts = pts2d.astype(int)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(img, hull, color)
    # Add an outline
    cv2.polylines(img, [hull], isClosed=True, color=(0, 0, 0), thickness=2)
    return img


def generate(args):
    out_root = Path(args.output)
    images_dir = out_root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Example object 3D keypoints (cube corners, size ~ 0.04 m)
    model_kpts = np.array([
        [0.02, 0.02, 0.0],
        [-0.02, 0.02, 0.0],
        [-0.02, -0.02, 0.0],
        [0.02, -0.02, 0.0],
        [0.02, 0.02, 0.04],
        [-0.02, 0.02, 0.04],
        [-0.02, -0.02, 0.04],
        [0.02, -0.02, 0.04]
    ], dtype=np.float32)

    # Save obj_models npz
    obj_models = {1: model_kpts}
    np.savez(out_root / 'obj_models.npz', obj_models=obj_models)

    # Camera intrinsics
    img_h, img_w = args.height, args.width
    fx = fy = args.fx
    cx = img_w / 2.0
    cy = img_h / 2.0
    K = create_camera_matrix(fx, fy, cx, cy)
    np.save(out_root / 'camera.npy', K)

    annotations_train = {}
    annotations_val = {}

    num_train = args.num_train
    num_val = args.num_val

    for i in range(num_train + num_val):
        image_id = f"{i:06d}"
        # Create blank background
        img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

        # Random pose
        R = random_rotation()
        z = random.uniform(0.4, 1.2)  # distance in meters
        t = np.array([random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), z], dtype=np.float32)

        # Project keypoints
        kpts2d = project_points(model_kpts, R, t, K)

        # Render object
        color = (int(random.uniform(50, 200)), int(random.uniform(50, 200)), int(random.uniform(50, 200)))
        img = render_simple_object(img, kpts2d, color=color)

        # Optionally add noise / texture
        if random.random() < 0.5:
            noise = (np.random.randn(img_h, img_w, 3) * 8).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Compute bbox
        x_min = float(np.min(kpts2d[:, 0]))
        x_max = float(np.max(kpts2d[:, 0]))
        y_min = float(np.min(kpts2d[:, 1]))
        y_max = float(np.max(kpts2d[:, 1]))

        # Clamp bbox
        x1 = max(0.0, x_min - 4.0)
        y1 = max(0.0, y_min - 4.0)
        x2 = min(img_w - 1.0, x_max + 4.0)
        y2 = min(img_h - 1.0, y_max + 4.0)

        # Save image
        img_path = images_dir / f"{image_id}.jpg"
        cv2.imwrite(str(img_path), img)

        ann_obj = {
            'category_id': 1,
            'bbox': [x1, y1, x2, y2],
            'keypoints': kpts2d.tolist(),
            'pose': {
                'rotation': R.tolist(),
                'translation': t.tolist()
            }
        }

        ann_entry = {
            'image_id': image_id,
            'objects': [ann_obj]
        }

        if i < num_train:
            annotations_train[image_id] = ann_entry
        else:
            annotations_val[image_id] = ann_entry

    # Save annotations files expected by dataset loader
    with open(out_root / 'annotations_train.json', 'w') as f:
        json.dump(annotations_train, f, indent=2)
    with open(out_root / 'annotations_val.json', 'w') as f:
        json.dump(annotations_val, f, indent=2)

    print(f"Generated dataset at {out_root}, train={num_train}, val={num_val}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./data/synthetic')
    parser.add_argument('--num-train', type=int, default=40)
    parser.add_argument('--num-val', type=int, default=10)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--fx', type=float, default=200.0)
    args = parser.parse_args()
    generate(args)
