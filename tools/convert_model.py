"""模型转换工具：从训练检查点导出仅包含 model_state_dict 的文件。"""
import argparse
import torch
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    torch.save(state, args.out)
    print(f"Saved converted model to {args.out}")

if __name__ == '__main__':
    main()
