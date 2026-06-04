"""简单评估脚本：对数据集运行推理并计算 ADD（示例）。"""
import argparse
import numpy as np
import cv2
from inference.predictor import Predictor
from train.metrics import add_error
from data.dataset import PoseDataset
from data.transforms import get_transforms
from config.config_loader import load_config
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--obj-models', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cam = np.eye(3)
    obj_models = np.load(args.obj_models, allow_pickle=True)['obj_models'].item()

    predictor = Predictor(model_path=args.model, camera_matrix=cam, obj_models=obj_models)

    transforms = get_transforms(cfg.dataset.to_dict(), split='val')
    ds = PoseDataset(args.data_root, split='val', transforms=transforms, num_keypoints=cfg.model.num_keypoints, image_size=(cfg.model.input_size, cfg.model.input_size))

    add_list = []
    for i in range(len(ds)):
        item = ds[i]
        img = (item['image'].permute(1,2,0).numpy()*255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_objs = item['poses']
        res = predictor.predict(img)
        # For simplicity, match first object
        if len(res) == 0 or len(gt_objs) == 0:
            continue
        pred = res[0]
        gt = gt_objs[0]
        model_points = obj_models[pred['cls']]
        err = add_error(pred['R'], pred['t'], gt['R'], gt['t'], model_points)
        add_list.append(err)

    print(f"Mean ADD: {np.mean(add_list) if add_list else float('nan')}")

if __name__ == '__main__':
    main()
