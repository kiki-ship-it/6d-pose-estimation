"""系统架构说明（简要）"""

整体流程：

```
输入图像 -> YOLO 检测 -> ROI 裁剪 -> 关键点检测 -> PnP 求解 -> 6D 姿态
```

- 检测器：ultralytics YOLOv8
- 关键点头：轻量级 CNN，输出 K 个热图并用 soft-argmax 提取像素坐标
- 求解器：OpenCV solvePnPRansac
