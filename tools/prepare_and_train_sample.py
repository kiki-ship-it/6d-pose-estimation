"""Prepare synthetic data and launch a quick training run (local)."""
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print('Running:', ' '.join(cmd))
    p = subprocess.Popen(cmd)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError('Command failed: ' + ' '.join(cmd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./data/synthetic')
    parser.add_argument('--config', type=str, default='config/config_small.yaml')
    parser.add_argument('--work-dir', type=str, default='./workdir/sample')
    args = parser.parse_args()

    # 1) Generate synthetic data
    run_cmd([sys.executable, 'data/generate_synthetic.py', '--output', args.data_root, '--num-train', '40', '--num-val', '10'])

    # 2) Convert example obj model into npz mapping (already saved by generator as obj_models.npz)

    # 3) Run training
    run_cmd([sys.executable, 'tools/train.py', '--config', args.config, '--data-root', args.data_root, '--work-dir', args.work_dir])

    print('Sample run completed. Check workdir for checkpoints and logs.')

if __name__ == '__main__':
    main()
