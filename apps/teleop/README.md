# apps/teleop — 视觉遥操（跨子系统）

视觉遥操（D455 + MediaPipe / HaMeR）→ 臂末端跟随 + 手 16-DOF 跟随。

## 当前内容

```
apps/teleop/
├── __init__.py
└── README.md
```

## 设计原则

1. **不动 submodule 代码** —— 直接 `sys.path` 引用 `Leap_Hand/python/gesture_mapping/` 下现成的纯函数模块
2. **本目录只做"跨子系统编排"** —— 多相机时间同步、按键分发、安全门
3. **demo 保留原入口** —— `Leap_Hand/python/gesture_mapping/demo_arm_teleop.py` 与 `demo_hamer3d.py` 仍是单一子系统的演示入口；本目录的运行脚本待 Phase A 完成 massage RobotV2 后再补

## 规划中的脚本（Phase A 之后再启动）

| 文件 | 作用 |
|------|------|
| `arm_teleop.py` | 等价于 `demo_arm_teleop.py` 的跨子系统入口 |
| `hand_teleop.py` | 等价于 `demo_hamer3d.py` 的跨子系统入口 |
| `bimanual_teleop.py` | 双手协调（更靠后） |

## 临时调用方式

Phase A 完成前, 直接复用 submodule 的 demo：

```bash
# 单臂视觉遥操 (需在另一终端跑 Arm-robot_VLA 的 mujoco_sim)
cd Leap_Hand/python
python gesture_mapping/demo_arm_teleop.py --port socket://localhost:5555

# 灵巧手手势映射
cd Leap_Hand/python
python gesture_mapping/demo_realtime.py [--drive]
```

## 依赖关系

```
apps/teleop/
   ├── Arm-robot_VLA/                          ← 机械臂通信层 (submodule)
   └── Leap_Hand/python/gesture_mapping/       ← 视觉遥操模块 (submodule)
```
