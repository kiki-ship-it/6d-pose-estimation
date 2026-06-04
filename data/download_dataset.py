"""Dataset download helper (示例)。"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ycb-video', choices=['ycb-video'])
    parser.add_argument('--data-root', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'ycb-video':
        print('请手动下载 YCB-Video 数据集并解压到指定目录。')
        print('参考: https://rse-lab.cs.washington.edu/projects/posecnn/datasets/')
    else:
        raise ValueError('Unsupported dataset')

if __name__ == '__main__':
    main()
