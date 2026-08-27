# TuinaDex

<div align="center">

**22-DOF 机械臂-灵巧手协同视觉遥操与具身智能（VLA）中医推拿系统**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware](https://img.shields.io/badge/Hardware-SocketCAN%20%7C%20Dynamixel%204Mbps-orange.svg)]()
[![Framework](https://img.shields.io/badge/Framework-Hugging%20Face%20LeRobot-yellow.svg)](https://github.com/huggingface/lerobot)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

[系统架构](#-系统架构) • [模块结构](#-模块结构) • [硬件准备](#-硬件准备) • [快速开始](#-快速开始) • [快捷键速查](#-遥操操作速查) • [LeRobot-接入规范](#-lerobot-具身智能接入) • [文档索引](#-文档索引)

</div>

---

## 📖 项目简介

**TuinaDex** 是面向中医推拿（Tuina Massage）场景构建的 22 自由度双硬件协同具身智能系统：

- **6-DOF 机械臂**：基于中大通 / Emm_V5 闭环步进电机，PC 通过 Linux 原生 **SocketCAN** 直连控制（500 kbps），提供宏观轨迹与姿态进近；
- **16-DOF 灵巧手**：集成 **LEAP Hand V1**，PC 通过 USB-TTL 串口直连 Dynamixel 舵机总线（4 Mbps），提供微观五指揉捏、抓捏与按压；
- **单目深度视觉协同遥操 (`Co_Teleop`)**：单台 RealSense D455 深度相机同时解算人手宏观位姿与五指 21 关节点几何，配备 1280x720 宽屏 HUD 仪表盘与分级安全看门狗；
- **LeRobot 具身智能闭环 (`Arm-robot_VLA`)**：无缝对接 Hugging Face LeRobot 数据采集格式，支持从遥操示教数据直接训练 SmolVLA / Diffusion Policy / ACT 策略模型。

---

## 🏗 系统架构

```text
                     【Intel RealSense D455 深度感知流】
                                     │
                    【MediaPipe 3D 手势关键点检测】
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
   【6-DOF 机械臂宏观位姿解耦】                      【16-DOF 灵巧手微观几何映射】
   • 刚体掌骨基准解算 (Wrist / MCPs)                 • 5 指 21 关节点向量夹角
   • 手腕空间位移 → 笛卡尔线速度 (vx, vy, vz)         • 指尖屈伸/侧摆 → 16 舵机目标弧度
   • 掌面倾斜偏角 → 虚拟摇杆角速度 (wx, wy, wz)       • 柔顺电流限制 + 1€ 滤波
            │                                                 │
            ▼                                                 ▼
   【四级安全看门狗 VisionWatchdog】                  【丢失渐进松开保护 relax_step】
   (OK / DECAY / STOP / ESTOP)                                │
            │                                                 │
            ▼                                                 ▼
   【6-DOF 闭环笛卡尔控制器 (SocketCAN)】            【LEAP Hand 实体驱动 (Dynamixel 4Mbps)】
            │                                                 │
            └────────────────────────┬────────────────────────┘
                                     │
                                     ▼
                    【LeRobot 数据集录制 / 策略闭环】
                    (22-DOF Joint State & Action @ 30Hz)
```

---

## 📂 模块结构

```text
TuinaDex/
├── Co_Teleop/                     # 机械臂-灵巧手 22-DOF 协同视觉遥操统一子系统
│   ├── pipeline/                  # 协同遥操主控管线 (unified_teleop.py)
│   ├── adapters/                  # 软硬件解耦适配层 (RealArmAdapter, LeapHandAdapter)
│   ├── safety/                    # 视觉丢失四级看门狗 (VisionWatchdog)
│   ├── calibration/               # 3D SVD 手眼标定向导与外参配置
│   ├── config/                    # 集中参数配置 (teleop_config.yaml)
│   ├── gui/                       # 1280x720 宽屏 HUD 监控仪表盘
│   └── tests/                     # 自动化单元测试套件 (55 项全通过)
│
├── Arm-robot_VLA/                 # 6-DOF 机械臂驱动、运动学算法与 LeRobot 适配子模块
│   ├── lerobot_robot_massage/     # LeRobot 机器人硬件实现类 (TuinaRobot)
│   │   └── zdt/                   # 6 轴 CAN 步进驱动、MDH 正逆运动学、DLS 阻尼与安全状态机
│   ├── configs/                   # 机械臂关节物理限位、减速比与训练超参数配置
│   ├── scripts/                   # 仿真测试 (simulation/)、启动脚本 (bringup/) 与控制工具
│   ├── docs/                      # 架构设计 (ARCHITECTURE.md)、CAN 协议与 LeRobot 接入规范
│   └── firmware/                  # 下位机/网关嵌入式固件源码 (C / FreeRTOS)
│
├── Leap_Hand/                     # 16-DOF 灵巧手驱动、MediaPipe 手势跟踪与关节映射子模块
│   ├── python/                    # 灵巧手 Python 核心驱动与视觉映射
│   │   ├── leap_hand_utils/       # Dynamixel SDK 封装 (4Mbps 串口) 与舵机限位管理
│   │   ├── gesture_mapping/       # MediaPipe 3D / HaMeR 3D 手势识别与 1€ 指尖滤波
│   │   └── tests/                 # 灵巧手单体测试与交互控制脚本
│   ├── ros2_module/               # ROS2 驱动包与话题/服务定义 (rclpy)
│   ├── CAD/                       # 灵巧手 3D 打印结构件与装配模型文件 (STEP / STL)
│   ├── useful_tools/              # 舵机 ID 配置、固件烧录与零位校准辅助工具
│   └── docs/                      # 硬件装配指南与电机电气特性说明
│
├── run_teleop.py                  # 顶层协同遥操一键启动入口
└── pyproject.toml                 # 顶层工程依赖与打包配置
```

---

## 🔌 硬件准备

| 设备 | 物理接口 | 接口参数 | 系统设备路径 |
| :--- | :--- | :--- | :--- |
| **6-DOF 机械臂** | USB-CAN 适配器 | SocketCAN, 波特率 500,000 bps | `can0` |
| **16-DOF LEAP Hand** | USB-TTL 串口 | Dynamixel Protocol 2.0, 4,000,000 bps | `/dev/ttyUSB0` |
| **手势遥操相机** | USB 3.0 | 深度流 640x480 @ 30fps | Intel RealSense D455 |
| **背部上帝视角相机** | USB 2.0/3.0 | RGB 流 640x480 @ 30fps (用于穴位识别与 VLA 观测) | `/dev/video0` (或指定索引) |

### 接口初始化

```bash
# 1. 激活机械臂 SocketCAN 接口
sudo ip link set can0 up type can bitrate 500000
ip -details link show can0

# 2. 配置灵巧手串口读写权限
sudo chmod 666 /dev/ttyUSB0
# 或永久加入 dialout 组: sudo usermod -aG dialout $USER
```

---

## 🚀 快速开始

### 1. 环境安装

推荐使用 Conda 管理 Python 运行环境（Python 3.10+）：

```bash
# 克隆仓库及所有子模块
git clone --recurse-submodules https://github.com/MacroBright/TuinaDex.git
cd TuinaDex

# 执行环境初始化脚本 (安装 editable 包与依赖)
bash scripts/setup_dev.sh
```

### 2. 运行 22-DOF 协同视觉遥操

确保 `can0` 与 `/dev/ttyUSB0` 已就绪，执行顶层入口：

```bash
# 启动真机 6-DOF 机械臂 + 16-DOF 灵巧手协同遥操 (默认 can0)
python run_teleop.py --iface can0

# 跳过启动确认提示直接进入
python run_teleop.py --iface can0 -y

# 无实体硬件时的调试模式 (仅渲染 1280x720 HUD 与手势追踪)
python run_teleop.py --no-drive
```

---

## 🎮 遥操操作速查

| 快捷键 | 功能 | 说明 |
| :---: | :--- | :--- |
| **`SPACE`** | **机械臂离合切换** | 暂停 / 恢复机械臂空间跟随（暂停时可自由调整操作员手势） |
| **`L`** | **灵巧手姿态锁定** | 锁定灵巧手当前抓捏角度并保持力矩；再次按下恢复视觉跟随 |
| **`1` / `2` / `3`** | **灵敏度档位切换** | `1` 档低速微调 (15 mm/s) \| `2` 档标准推拿 (40 mm/s) \| `3` 档高速换位 |
| **`R`** | **准备姿态 (READY)** | 机械臂安全平缓运动至按摩作业准备姿态，灵巧手回全开位 |
| **`H`** | **初始复位 (HOME)** | 机械臂垂直回零复位，灵巧手回全开位 |
| **`Z`** | **视觉基准归零** | 将当前人手空间位置重设为机械臂静止原点 |
| **`K`** | **灵巧手零位校准** | 读取当前舵机绝对偏置并写入校准参数表 |
| **`Q` / `ESC`** | **安全停机退出** | 机械臂安全断电、灵巧手释放扭矩，平稳退出程序 |

---

## 🤖 LeRobot 具身智能接入

TuinaDex 提供标准接口将 22-DOF 遥操数据导出为 Hugging Face LeRobot 标准数据集（Parquet + MP4），并支持下游直接调用：

- **数据特征空间**：
  - `observation.state`: `(22,) float32`（机械臂 6 轴 + 灵巧手 16 轴真实关节弧度，单位 $\text{rad}$）
  - `action`: `(22,) float32`（22 轴下一周期目标控制量，单位 $\text{rad}$）
  - `observation.images.overhead_cam`: `(3, 480, 640) uint8`（人体背部上帝视角相机 RGB 图像）
- **数据清洗保障**：在操作员按下 `SPACE` 暂停、`L` 锁定或手部丢失看门狗触发期间，录制器自动暂停写入，杜绝“原地发呆”无意义数据。

详见完整接入规范与实现代码：  
👉 **[docs/tuinadex_to_lerobot.md](./docs/tuinadex_to_lerobot.md)**

---

## 📚 文档索引

- **协同遥操子系统教程**：[Co_Teleop/README.md](./Co_Teleop/README.md)
- **LeRobot 接入与对接规范**：[docs/tuinadex_to_lerobot.md](./docs/tuinadex_to_lerobot.md)
- **机械臂底层 CAN 驱动协议**：[docs/HARDWARE_CAN_SPEC.md](./docs/HARDWARE_CAN_SPEC.md)
- **系统架构设计记录 (ADR)**：[Arm-robot_VLA (GitHub)](https://github.com/MacroBright/Arm-robot_VLA/blob/feat/arm-visual-teleop/CLAUDE.md)
- **Docker 多容器部署指南**：[docker/README.md](./docker/README.md)

---

## 📄 License

- 顶层工程与 `Co_Teleop` 子系统：[Apache-2.0](./LICENSE)
- `Arm-robot_VLA` 机械臂模块：[Apache-2.0](https://github.com/MacroBright/Arm-robot_VLA/blob/feat/arm-visual-teleop/LICENSE)
- `Leap_Hand` 灵巧手模块：[CC BY-NC 4.0](https://github.com/MacroBright/Leap_hand/blob/feat/arm-visual-teleop/LICENSE)（非商业使用限制）
