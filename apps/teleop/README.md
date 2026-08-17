# apps/teleop — 视觉遥操（跨子系统）

> **跨子系统应用层**：视觉遥操（D455 + MediaPipe/HaMeR）→ 臂末端跟随 + 手 16-DOF 跟随。

## 包含（占位）

- `arm_teleop.py` —— 复用 `Leap_Hand/python/gesture_mapping/demo_arm_teleop.py` 的视觉遥操逻辑
- `hand_teleop.py` —— 复用 `Leap_Hand/python/gesture_mapping/demo_hamer3d.py` 的 HaMeR 3D → LEAP 16-DOF 关节映射
- `bimanual_teleop.py` —— 双手/单臂多模态协调（未来）

## 设计原则

1. **不动 submodule 代码**——直接通过 `sys.path` 引用 `gesture_mapping.*` 的纯函数模块
2. **本目录专注"跨子系统编排"**——视觉框架的多相机时间同步、按键分发、安全门
3. **复用现有 demo**——`demo_arm_teleop.py` / `demo_hamer3d.py` 保留作为单一子系统 demo 入口

## 运行

```bash
# 单一子系统（臂）—— 等价于 Leap_Hand/python/gesture_mapping/demo_arm_teleop.py
python apps/teleop/arm_teleop.py --port socket://localhost:5555

# 单一子系统（手）—— 等价于 Leap_Hand/python/gesture_mapping/demo_hamer3d.py
python apps/teleop/hand_teleop.py

# 双手协调（占位）
python apps/teleop/bimanual_teleop.py
```

> ⚠ **当前阶段**：占位骨架。Phase A 完成 massage RobotV2 后再启动。

## 依赖关系

```
apps/teleop/
   ├── Arm-robot_VLA/                  ← 机械臂通信层（submodule）
   └── Leap_Hand/python/gesture_mapping/  ← 视觉遥操模块（submodule）
```