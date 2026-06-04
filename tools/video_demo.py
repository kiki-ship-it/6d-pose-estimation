"""视频与摄像头实时演示脚本。"""
import argparse
import cv2
import numpy as np
from inference.predictor import Predictor
from inference.visualizer import visualize_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--camera-matrix', type=str, default=None)
    parser.add_argument('--obj-models', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    camera_matrix = np.load(args.camera_matrix) if args.camera_matrix else np.eye(3)
    obj_models = None
    if args.obj_models:
        obj_models = np.load(args.obj_models, allow_pickle=True)['obj_models'].item()

    predictor = Predictor(model_path=args.model, camera_matrix=camera_matrix, obj_models=obj_models)

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = predictor.predict(rgb)
        vis = visualize_results(frame, results, camera_matrix, None)
        cv2.imshow('6D Pose Demo', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
