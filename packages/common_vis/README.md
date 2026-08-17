# packages/common_vis — 跨子系统共享视觉工具（占位骨架）

跨 Arm / Hand / 顶层应用共享的视觉工具（图像处理、相机标定、关键点检测等）。

> 当前阶段: **不复制任何 submodule 代码**——通过 `sys.path` 让 submodule 内部模块可被顶层引用。本目录的 `__init__.py` 只声明包存在。

## 候选复用源（暂不抽取, 走 sys.path）

| 来源 | 可复用内容 |
|------|-----------|
| `Arm-robot_VLA/scripts/mujoco_sim.py` | MuJoCo 离屏渲染 + 共享内存相机帧 |
| `Arm-robot_VLA/scripts/camera_server.py` | 共享内存帧生产者 |
| `Leap_Hand/python/gesture_mapping/camera.py` | RealSense D455 抽象 (`open_realsense`) |
| `Leap_Hand/python/gesture_mapping/hand_tracker.py` | MediaPipe 21kp 检测 |
| `Leap_Hand/python/gesture_mapping/hamer_3d.py` | HaMeR 3D MANO 推理 |

## 设计原则

1. **不复制代码** —— 直接通过 `sys.path` 注入 submodule 内部模块
2. **本目录放"跨项目共用"的视觉辅助** —— 例如共享内存相机帧协议（Arm 已用、Hand 也需要时）、通用深度反投影
3. **跨项目统一相机命名约定** —— `cam_top` / `cam_wrist` / `cam_teleop`（见 [Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md §3.4](../../Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md)）

## 典型用法（未来）

```python
# 顶层 apps/teleop/bimanual_teleop.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Leap_Hand" / "python"))
from gesture_mapping.camera import open_realsense   # 复用 Hand 端 RealSense 抽象
```