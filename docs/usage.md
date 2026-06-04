"""使用说明"""

# 使用示例

1. 下载数据集（参见 docs/dataset_setup.md）
2. 训练示例：

```bash
python tools/train.py --config config/config.yaml --data-root ./data
```

3. 单图演示：

```bash
python tools/demo.py --model weights/best_model.pth --image test.jpg --camera camera.npy --obj-models obj_models.npz
```
