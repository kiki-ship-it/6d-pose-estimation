"""命令行演示：对单张图像运行推理并保存可视化结果。"""
import argparse
import numpy as np
import cv2
from inference.predictor import Predictor
from inference.visualizer import visualize_results
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='关键点模型权重路径 (.pth)')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--camera', type=str, default=None, help='numpy 文件，包含 camera_matrix (3x3)')
    parser.add_argument('--obj-models', type=str, default=None, help='npz 文件，包含 obj_models 字典')
    parser.add_argument('--out', type=str, default='out.jpg')
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    camera_matrix = None
    dist_coeffs = None
    if args.camera:
        camera_matrix = np.load(args.camera)

    obj_models = None
    if args.obj_models:
        data = np.load(args.obj_models, allow_pickle=True)
        obj_models = data['obj_models'].item()

    predictor = Predictor(model_path=args.model, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, obj_models=obj_models)
    results = predictor.predict(image)

    vis = visualize_results(image[:, :, ::-1], results, camera_matrix if camera_matrix is not None else np.eye(3), dist_coeffs)
    out_path = Path(args.out)
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
