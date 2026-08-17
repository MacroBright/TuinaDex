# packages/common_math — 跨子系统共享数学工具（占位骨架）

跨 Arm / Hand / 顶层应用共享的数学工具（旋转、IK、四元数、滤波等）。

> 当前阶段: **不复制任何 submodule 代码**——通过 `sys.path` 让 submodule 内部模块可被顶层引用。本目录的 `__init__.py` 只声明包存在，真正的共用代码待 Phase A 启动后按调用频率决定是否上提。

## 候选复用源（暂不抽取, 走 sys.path）

| 来源 | 可复用内容 |
|------|-----------|
| `Arm-robot_VLA/scripts/remote_semantics.py` | `remote_event` 7 参 → `(v_lin, j4, j5, j6)` 纯函数 |
| `Leap_Hand/python/gesture_mapping/wrist_tracker.py` | `backproject` / `palm_basis` / `quat_to_rot` / `rot_error_angvel` |
| `Leap_Hand/python/gesture_mapping/handeye_calib.py` | `apply_rotation` / `solve_handeye` |
| `Leap_Hand/python/gesture_mapping/filter.py` | `OneEuroFilter` |
| `Leap_Hand/python/leap_hand_utils/leap_hand_utils.py` | `LEAPsim_to_LEAPhand` / `angle_safety_clip` |

## 设计原则

1. **不复制代码** —— 通过 `sys.path` 让 submodule 内部模块可被顶层引用
2. **本目录仅放"跨子系统共用"的辅助** —— 例如 4×4 矩阵 ↔ wxyz 四元数转换、通用滤波这类两边都需要的底层函数
3. **不强行统一命名空间** —— `gesture_mapping` / `lerobot_robot_massage` 子模块内部命名保持不变

## 典型用法（未来）

```python
# 顶层 apps/massage/massage_robot_v2.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Leap_Hand" / "python"))
from gesture_mapping.wrist_tracker import WristTracker   # 复用 Hand 端代码
```