# apps/massage — 22-DOF 机械臂+灵巧手按摩应用

> **跨子系统应用层**：本目录代码同时引用 `Arm-robot_VLA` 与 `Leap_Hand` 两个 submodule，
> 解决"单一文件跨两个子系统"的整合开发痛点。

## 包含

- `massage_robot_v2.py` —— LeRobot `Robot` 子类 (22 DOF = 6 臂 + 16 手)
- `data_collector.py` —— 22 DOF 真机数据采集（LeRobotDataset 写入）
- `safety_gate.py` —— 跨子系统安全协调（e_stop 一键停两个子系统）

## 设计原则

1. **不复制任何 submodule 代码**——通过 Python `sys.path` 引用 `lerobot_robot_massage`
   与 `leap_hand_utils`。
2. **本目录的代码可独立测试**——`tests/` 覆盖跨子系统交互。
3. **submodule 升级不影响本目录**——除非新功能需要，否则无需修改。

## 依赖关系

```
apps/massage/        ← 顶层应用（跨子系统）
   ├── Arm-robot_VLA/lerobot_robot_massage/    (submodule)
   │   ├── MassageRobot (6-DOF)
   │   ├── serial_protocol
   │   └── zdt/controller.py
   │
   └── Leap_Hand/python/leap_hand_utils/      (submodule)
       ├── DynamixelClient (16-DOF)
       └── leap_hand_utils (角度约定)
```

## 计划（来自 Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md）

- Phase A：仿真验证数据管线（无新硬件风险）
- Phase B：真机通信打通（STM32 固件 ZDT 化 + LEAP Hand 上电）
- Phase C：真机数据采集（每穴位 ≥10 episodes）
- Phase D：训练与部署（SmolVLA 22-DOF action space）

> ⚠ **当前阶段**：占位骨架。Phase A 待启动。

## 运行

```bash
# 仿真模式（mujoco_sim.py TCP 5555 + 手部真实 USB）
python apps/massage/massage_robot_v2.py --mode sim

# 真机模式
python apps/massage/massage_robot_v2.py --mode real --arm-port /dev/ttyUSB0 --hand-port /dev/ttyUSB1
```

> 完整实现见 `Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md` §3.3。