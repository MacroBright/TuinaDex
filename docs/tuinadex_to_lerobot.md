# TuinaDex 22-DOF 中医推拿机械臂-灵巧手 LeRobot 接入与数据对接技术规范 (tuinadex_to_lerobot)

> **文档版本**: v1.1.0  
> **适用对象**: 上游视觉遥操开发团队 (`Co_Teleop`)、下游 LeRobot / VLA 具身智能算法团队 (`Arm-robot_VLA`)  
> **系统硬件**: 6-DOF 闭环步进机械臂 (SocketCAN, 500kbps) + 16-DOF LEAP Hand 灵巧手 (Dynamixel 4Mbps) + 背部上帝视角相机 (Acupoint RGB) + 操作员手势遥控相机 (RealSense D455)

---

## 目录
1. [系统总体架构与数据流](#1-系统总体架构与数据流)
2. [标准数据契约与特征空间定义 (Schema & Spec)](#2-标准数据契约与特征空间定义-schema--spec)
3. [电机通信协议与 LeRobot 四大生命周期方法对接解析](#3-电机通信协议与-lerobot-四大生命周期方法对接解析)
4. [TuinaRobot 在 LeRobot 官方框架中的注册与 CLI 调用指南](#4-tuinarobot-在-lerobot-官方框架中的注册与-cli-调用指南)
5. [模式一：快速产数对接方案 (Co_Teleop 内置 LeRobotDataset Recorder)](#5-模式一快速产数对接方案-co_teleop-内置-lerobotdataset-recorder)
6. [模式二：标准 LeRobot 类封装代码 (TuinaRobot & TuinaVisualTeleop)](#6-模式二标准-lerobot-类封装代码-tuinarobot--tuinavisualteleop)
7. [人体背部上帝视角相机与穴位检测模型协同](#7-人体背部上帝视角相机与穴位检测模型协同)
8. [数据清洗准则：离合与看门狗过滤规范](#8-数据清洗准则离合与看门狗过滤规范)
9. [下游 VLA / Policy 训练与安全推理评估指南](#9-下游-vla--policy-训练与安全推理评估指南)
10. [接口对接常见问题 (FAQ) 与联调排错](#10-接口对接常见问题-faq-与联调排错)

---

## 1. 系统总体架构与数据流

整体系统分为 **数据采集态（Teleoperation Collection）** 与 **策略评估/推理态（Autonomous Evaluation）** 两个阶段：

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           1. 采集态 (Data Collection @ 30Hz)                      │
│                                                                                  │
│   【操作员手势】                          【推拿作业区】                           │
│   RealSense D455 3D关键点              背部上帝视角相机 (Overhead RGB)             │
│        │                                        │                                │
│        ▼                                        ▼                                │
│   【Co_Teleop 遥操管线】                   【穴位检测 / 视觉观测】                │
│   • 机械臂 6-DOF 刚体位姿解算              • 捕捉背部推拿区域图像                 │
│   • 灵巧手 16-DOF 几何映射                 • 提取经络穴位关键点                   │
│        │                                        │                                │
│        ├────────────────────────────────────────┴─────────────────┐              │
│        │                                                          ▼              │
│        ▼                                                 【LeRobotDataset 写入器】│
│   【实体硬件执行】                                       • 过滤静止/离合帧        │
│   • 机械臂 SocketCAN 闭环 (0xFD)                         • 写入 22-DOF State    │
│   • LEAP Hand 4Mbps (0~16 弧度)                          • 写入 22-DOF Action   │
│                                                          • 写入 Overhead MP4    │
└───────────────────────────────────────────────────────────────────┬──────────────┘
                                                                    │ 生成 HF 数据集
                                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           2. 训练态 (Policy Training)                            │
│                                                                                  │
│   LeRobot 训练管线 (SmolVLA-450M / Diffusion Policy / ACT)                        │
│   • Input: observation.images.overhead_cam + "按揉大椎穴" (Language Prompt)      │
│   • Input State: observation.state (22-dim joint angles)                         │
│   • Target: action chunk (22-dim target joint angles sequence)                   │
└───────────────────────────────────────────────────────────────────┬──────────────┘
                                                                    │ 输出权重
                                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           3. 推理评估态 (Autonomous Inference)                   │
│                                                                                  │
│   Overhead Camera ──► Policy.select_action(obs) ──► 22-DOF Action Chunk          │
│                                                            │                     │
│                                                            ▼                     │
│                                                 【TuinaSafetyLayer 安全包络】    │
│                                                 • 关节软限位 Clamping            │
│                                                 • 单步最大跳变限幅 (max_dq)       │
│                                                 • 机械臂-灵巧手实体硬件执行      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 标准数据契约与特征空间定义 (Schema & Spec)

所有数据在进入 LeRobot 数据集或由模型输出时，**严格采用国际标准单位制 (SI Units: 弧度 Radian)**。

### 2.1 观测空间 (Observation Space)

| 特征键名 (`features` key) | 维度 / 类型 | 范围 / 说明 |
| :--- | :--- | :--- |
| `observation.state` | `(22,) float32` | 22 自由度当前关节状态（前 6 维机械臂，后 16 维灵巧手，单位：$\text{rad}$） |
| `observation.images.overhead_cam` | `(3, H, W) uint8` | 人体背部上帝视角相机 RGB 图像（推荐 480x640 或 720x1280） |
| `observation.images.wrist_cam` *(可选)* | `(3, H, W) uint8` | 机械臂手腕微距穴位接触视角相机（若有） |

### 2.2 动作空间 (Action Space)

| 特征键名 (`features` key) | 维度 / 类型 | 范围 / 说明 |
| :--- | :--- | :--- |
| `action` | `(22,) float32` | 22 自由度下一控制周期的**目标关节绝对弧度 (rad)** |

### 2.3 22 个维度的关节映射详细定义

```text
Index 00 ~ 05: 【6-DOF 机械臂关节】
  - Index 00: J1 基座水平回转 (Base Yaw)        [-0.017, 6.283] rad (0° ~ 360°)
  - Index 01: J2 大臂主俯仰 (Shoulder Pitch)     [-0.017, 2.618] rad (0° ~ 150°)
  - Index 02: J3 小臂肘部俯仰 (Elbow Pitch)      [-0.017, 2.094] rad (0° ~ 120°)
  - Index 03: J4 小臂腕部滚转 1 (Wrist Roll 1)   [-1.571, 1.571] rad (-90° ~ +90°)
  - Index 04: J5 腕部俯仰 (Wrist Pitch)         [-0.017, 3.142] rad (0° ~ 180°)
  - Index 05: J6 末端工具自转 (Wrist Roll 2)     [-0.017, 6.283] rad (0° ~ 360°)

Index 06 ~ 21: 【16-DOF LEAP Hand 灵巧手关节 (右手机械手)】
  - Index 06 ~ 09: 食指 (Index Finger)    [侧摆 MCP_yaw, 掌指 MCP_pitch, 近指 PIP, 远指 DIP]
  - Index 10 ~ 13: 中指 (Middle Finger)   [侧摆 MCP_yaw, 掌指 MCP_pitch, 近指 PIP, 远指 DIP]
  - Index 14 ~ 17: 无名指 (Ring Finger)   [侧摆 MCP_yaw, 掌指 MCP_pitch, 近指 PIP, 远指 DIP]
  - Index 18 ~ 21: 大拇指 (Thumb)        [基底 MCP_roll, 掌骨 CMC, 近指 MCP_pitch, 远指 IP]
```

> **注意（灵巧手符号约定）**：
> `LeapHandAdapter` 内部包含舵机正反向符号表 `JOINT_DIR`。交付给 LeRobot 的弧度值统一为**物理弯曲弧度**（0.0 表示全开位，正值表示弯曲揉捏），避免底层硬件安装方向干扰策略泛化性。

---

## 3. 电机通信协议与 LeRobot 四大生命周期方法对接解析

根据 LeRobot 官方《自带硬件 (Bring Your Own Hardware, BYOH)》教程，针对具有异构通信链路（CAN 总线 + 高速串口）的机器人，官方推荐采用 **路径 A：自定义 `Robot` 派生类** 封装硬件通信。

### 3.1 通信链路与协议差异

| 硬件子系统 | 物理总线 | 传输协议 | 关键协议指令/报文 | 现有工程适配模块 |
| :--- | :--- | :--- | :--- | :--- |
| **6-DOF 机械臂** | SocketCAN (`can0`) | ZDT 二进制 CAN 帧 (500kbps) | `0xF3` 使能、`0xFD` 脉冲驱动、`0x36` 角度读取、`0xFE` 广播急停 | `lerobot_robot_massage.zdt` / `RealArmAdapter` |
| **16-DOF 灵巧手** | USB-TTL 串口 (`/dev/ttyUSB0`) | Dynamixel Protocol 2.0 (4Mbps) | `GroupSyncRead` (批量读角度)、`GroupSyncWrite` (批量写弧度/限流) | `Co_Teleop.adapters.LeapHandAdapter` |

---

### 3.2 四大核心生命周期方法映射表

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       LeRobot 硬件生命周期与协议动作映射                                     │
├─────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┤
│ LeRobot 标准生命周期    │ 本项目底层通信协议的具体行为                                                      │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ 1. connect()            │ • 机械臂: 打开 can0，扫描枚举 0x02~0x07 电机，显式确认重力关节并发送 0xF3 使能扭矩│
│                         │ • 灵巧手: 打开 /dev/ttyUSB0 (4Mbps)，校验 16 舵机 Ping 状态，加载限位并使能扭矩   │
│                         │ • 相机: 启动背部上帝视角相机与手势相机捕获线程                                    │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ 2. capture_observation()│ • 机械臂: 发送 0x36 报文查询 6 轴编码器角度 (度)，转为标准弧度 (rad)              │
│                         │ • 灵巧手: 调用 GroupSyncRead 批量读取 16 轴实时弧度 (rad)                         │
│                         │ • 聚合: 拼接为 22 维 float32 数组 observation.state 并附加上帝视角 RGB 图像       │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ 3. send_action(action)  │ • 拆分: action[:6] 归机械臂，action[6:] 归灵巧手                                  │
│                         │ • 机械臂: 弧度转度 -> 软限位 Clamping -> 发送 0xFD 脉冲/速度报文驱动 6 轴电机     │
│                         │ • 灵巧手: 物理限位 Clamping -> GroupSyncWrite 批量下发 16 轴目标弧度与力矩保护    │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ 4. disconnect()         │ • 机械臂: 发送 0xFE/0xF3 广播急停并断电失能，安全关闭 SocketCAN 句柄              │
│                         │ • 灵巧手: 释放 16 轴舵机力矩，安全关闭串口句柄                                    │
│                         │ • 相机: 释放视频流采集资源                                                        │
└─────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. TuinaRobot 在 LeRobot 官方框架中的注册与 CLI 调用指南

为了让下游同事能够直接使用 LeRobot 官方命令行工具（如 `lerobot-record`、`lerobot-eval`）无缝调用本项目硬件，需将 `TuinaRobot` 注册至 LeRobot 注册表中。

### 4.1 注册方法（在下游环境中）

在 `lerobot/robots/__init__.py` 或项目入口中进行声明：

```python
# lerobot/robots/__init__.py (或本地扩展入口)
from Arm_robot_VLA.lerobot_robot_massage.tuina_robot import TuinaRobot, TuinaRobotConfig

# 注册配置类与机器人实现类
ROBOT_CONFIG_MAPPING["tuina_massage_robot"] = TuinaRobotConfig
ROBOT_MAPPING["tuina_massage_robot"] = TuinaRobot
```

### 4.2 LeRobot 官方 CLI 一键运行示例

```bash
# 1. 检查并激活硬件接口权限
sudo ip link set can0 up type can bitrate 500000
sudo chmod 666 /dev/ttyUSB0

# 2. 使用 LeRobot 官方标准命令进行推拿数据采集
python -m lerobot.scripts.record \
    --robot.type tuina_massage_robot \
    --robot.can_iface can0 \
    --robot.hand_port /dev/ttyUSB0 \
    --repo-id tuina_massage_dataset \
    --fps 30 \
    --num-episodes 50

# 3. 使用训练好的 SmolVLA / ACT 策略进行自主推拿闭环评估
python -m lerobot.scripts.eval \
    --robot.type tuina_massage_robot \
    --policy.path ./outputs/train/tuina_smolvla \
    --num-episodes 10
```

---

## 5. 模式一：快速产数对接方案 (Co_Teleop 内置 LeRobotDataset Recorder)

此方案直接在 `Co_Teleop/pipeline/unified_teleop.py` 中挂载录制器，**完全保留 1280x720 宽屏 HUD、手部骨骼显示、档位切换与即时离合保护**。

### 5.1 录制集成代码示例 (`Co_Teleop/pipeline/lerobot_recorder.py`)

```python
"""Co_Teleop 专用 LeRobot 数据集导出器 (基于 Hugging Face lerobot.common.datasets.lerobot_dataset)."""
from pathlib import Path
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

class TuinaEpisodeRecorder:
    def __init__(self, repo_id: str = "tuina_massage_22dof", fps: int = 30, root: Path = None):
        self.fps = fps
        self.repo_id = repo_id
        self.root = root or Path("./datasets/lerobot_tuina")
        
        # 定义 22-DOF 特征空间
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (22,),
                "names": [
                    "arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5", "arm_j6",
                    "hand_idx_yaw", "hand_idx_mcp", "hand_idx_pip", "hand_idx_dip",
                    "hand_mid_yaw", "hand_mid_mcp", "hand_mid_pip", "hand_mid_dip",
                    "hand_rng_yaw", "hand_rng_mcp", "hand_rng_pip", "hand_rng_dip",
                    "hand_thb_rot", "hand_thb_cmc", "hand_thb_mcp", "hand_thb_ip",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (22,),
                "names": [
                    "cmd_arm_j1", "cmd_arm_j2", "cmd_arm_j3", "cmd_arm_j4", "cmd_arm_j5", "cmd_arm_j6",
                    "cmd_hand_idx_yaw", "cmd_hand_idx_mcp", "cmd_hand_idx_pip", "cmd_hand_idx_dip",
                    "cmd_hand_mid_yaw", "cmd_hand_mid_mcp", "cmd_hand_mid_pip", "cmd_hand_mid_dip",
                    "cmd_hand_rng_yaw", "cmd_hand_rng_mcp", "cmd_hand_rng_pip", "cmd_hand_rng_dip",
                    "cmd_hand_thb_rot", "cmd_hand_thb_cmc", "cmd_hand_thb_mcp", "cmd_hand_thb_ip",
                ],
            },
            "observation.images.overhead_cam": {
                "dtype": "video",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
        }
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.root,
            features=features,
            use_videos=True,
        )
        self.recording = False

    def start_episode(self):
        """开始一个新的推拿动作 Episode."""
        self.recording = True

    def add_frame(self, overhead_bgr: np.ndarray, arm_deg: list, hand_rad: np.ndarray,
                  cmd_arm_deg: list, cmd_hand_rad: np.ndarray):
        """写入单帧数据 (自动进行 deg -> rad 转换)."""
        if not self.recording:
            return

        # 1. 机械臂角度转标准弧度
        arm_rad = np.deg2rad(np.asarray(arm_deg, dtype=np.float32))
        cmd_arm_rad = np.deg2rad(np.asarray(cmd_arm_deg, dtype=np.float32))
        
        # 2. 拼装 22 维 State 与 Action
        state_22 = np.concatenate([arm_rad, np.asarray(hand_rad, dtype=np.float32)])
        action_22 = np.concatenate([cmd_arm_rad, np.asarray(cmd_hand_rad, dtype=np.float32)])

        # 3. 图像转 RGB (LeRobot 约定通道顺序)
        overhead_rgb = np.transpose(overhead_bgr[:, :, ::-1], (2, 0, 1)) # (H,W,C) -> (C,H,W)

        frame = {
            "observation.state": state_22,
            "action": action_22,
            "observation.images.overhead_cam": overhead_rgb,
        }
        self.dataset.add_frame(frame)

    def end_episode(self, task_desc: str = "中医背部穴位推拿按揉"):
        """结束当前 Episode 并写入元数据."""
        if self.recording:
            self.dataset.save_episode(task=task_desc)
            self.recording = False

    def finish(self):
        """结束整批采集，合并并刷新元数据 (Consolidate)."""
        self.dataset.consolidate()
```

---

## 6. 模式二：标准 LeRobot 类封装代码 (TuinaRobot & TuinaVisualTeleop)

```python
"""Arm-robot_VLA/lerobot_robot_massage/tuina_robot.py"""
import logging
from dataclasses import dataclass, field
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig

# 引入已验证的底层通信协议与适配器
from Co_Teleop.adapters.arm_adapter import RealArmAdapter
from Co_Teleop.adapters.hand_adapter import LeapHandAdapter
from lerobot_robot_massage.zdt.config import ZdtConfig
from lerobot_robot_massage.zdt.controller import ZdtController

logger = logging.getLogger(__name__)


@dataclass
class TuinaRobotConfig(RobotConfig):
    name: str = "tuina_massage_robot"
    # 1. 机械臂 SocketCAN 通信参数
    can_iface: str = "can0"
    can_bitrate: int = 500000
    gravity_confirm: bool = True
    
    # 2. 灵巧手 Dynamixel 串口通信参数
    hand_port: str = "/dev/ttyUSB0"
    hand_baudrate: int = 4000000
    
    # 3. 控制频率与相机配置
    control_freq_hz: int = 30
    cameras: dict = field(default_factory=lambda: {
        "overhead_cam": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
        }
    })


class TuinaRobot(Robot):
    """22-DOF 机械臂-灵巧手 LeRobot 硬件实现类."""
    config_class = TuinaRobotConfig

    def __init__(self, config: TuinaRobotConfig):
        super().__init__(config)
        self.config = config
        
        # 初始化相机
        self._cameras = make_cameras_from_configs(config.cameras)
        
        # 1. 初始化机械臂 CAN 通信协议控制器 (ZdtController)
        zdt_cfg = ZdtConfig(channel=config.can_iface, bitrate=config.can_bitrate)
        self._arm_ctrl = ZdtController(zdt_cfg)
        self._arm = RealArmAdapter(self._arm_ctrl)
        
        # 2. 初始化灵巧手 Dynamixel 通信适配器 (LeapHandAdapter)
        self._hand = LeapHandAdapter()

    # ------------------------------------------------------------------
    # 1. 硬件连接与通信握手
    # ------------------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        # (1) 机械臂 CAN 通信握手与使能
        self._arm.connect()                                # 扫描总线 6 轴电机
        if self.config.gravity_confirm:
            self._arm.arm(gravity_confirmed=True)          # 发送 0xF3 使能扭矩
            self._arm.enter_teleop()
        logger.info("机械臂 SocketCAN (%s) 已握手并使能", self.config.can_iface)

        # (2) 灵巧手串口通信握手
        self._hand.connect(port=self.config.hand_port)     # 打开 4Mbps 串口握手 16 轴舵机
        logger.info("灵巧手 Dynamixel 串口 (%s) 已连接", self.config.hand_port)

        # (3) 相机连接
        for name, cam in self._cameras.items():
            cam.connect()
            logger.info("相机 '%s' 已就绪", name)

    # ------------------------------------------------------------------
    # 2. 读取电机状态 (Observation 采集)
    # ------------------------------------------------------------------
    def capture_observation(self) -> dict[str, np.ndarray]:
        obs = {}
        # (1) 读视觉图像
        for name, cam in self._cameras.items():
            obs[name] = cam.async_read()

        # (2) 读取机械臂 6 轴实时角度 (底层 0x36 报文，度转弧度)
        arm_deg = self._arm.get_joint_state().q
        arm_rad = np.deg2rad(np.asarray(arm_deg, dtype=np.float32))

        # (3) 读取灵巧手 16 轴当前弯曲角度 (底层 SyncRead，已经是弧度)
        hand_rad = self._hand.get_current_angles().astype(np.float32)

        # 拼装为 22 维连续状态向量
        obs["observation.state"] = np.concatenate([arm_rad, hand_rad])
        return obs

    # ------------------------------------------------------------------
    # 3. 下发电机控制指令 (Action 执行)
    # ------------------------------------------------------------------
    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        cmd_arm_rad = action[:6]
        cmd_hand_rad = action[6:]

        # (1) 机械臂执行: 弧度转度 -> 软限位 Clamping -> 发送 0xFD 脉冲命令
        cmd_arm_deg = np.rad2deg(cmd_arm_rad).tolist()
        self._arm._ctrl.set_joints(cmd_arm_deg)

        # (2) 灵巧手执行: 物理限位 Clamping -> 发送 SyncWrite 舵机目标弧度
        self._hand.set_angles(cmd_hand_rad)

    # ------------------------------------------------------------------
    # 4. 安全断电与释放资源
    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        if not self.is_connected:
            return

        try:
            self._arm.disconnect()     # CAN 广播急停 / 0xF3 失能扭矩
            self._hand.disconnect()    # 释放 16 轴舵机扭矩
        finally:
            for cam in self._cameras.values():
                cam.disconnect()
        logger.info("TuinaRobot 硬件资源已完全释放")
```

---

## 7. 人体背部上帝视角相机与穴位检测模型协同

### 7.1 上帝视角相机在系统中的双重作用
1. **宏观导航与穴位定位**：背部全局 RGB 图像输入至前置 **穴位关键点检测模型**，定位出“大椎穴”、“肩井穴”、“肺俞穴”等三维空间坐标（相对于机器人基座标系 $T_{\text{base}}^{\text{acupoint}}$）。
2. **闭环策略微调（VLA Policy）**：在接近目标穴位后，上帝视角相机图像作为 SmolVLA / Diffusion Policy 的 `observation.images.overhead_cam`，输出连续的 22-DOF 揉捏/滚法/推法轨迹。

### 7.2 相机标定与安装规范
- **分辨率与帧率**：统一锁定在 **640x480 @ 30fps**（兼顾推理速度与特征清晰度）。
- **坐标系对齐**：相机安装于推拿床正上方（高度 1.2m~1.5m），需完成相机的眼在手外（Eye-to-Base）标定，将穴位检测结果直接投影到机械臂基座坐标系。

---

## 8. 数据清洗准则：离合与看门狗过滤规范

高质量的数据集是模仿学习成功的基石。在数据采集过程中，必须执行以下**清洗过滤机制**：

```text
                  【采集单帧数据】
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ 判断 1: 机械臂是否被按 SPACE 暂停? │ ──是──► 【丢弃该帧，不写入 Dataset】
        └────────────────┬─────────────────┘
                         │ 否
                         ▼
        ┌──────────────────────────────────┐
        │ 判断 2: 灵巧手是否被按 L 锁定?   │ ──是──► 【丢弃该帧，不写入 Dataset】
        └────────────────┬─────────────────┘
                         │ 否
                         ▼
        ┌──────────────────────────────────┐
        │ 判断 3: 看门狗处于 STOP/ESTOP?   │ ──是──► 【丢弃该帧，并准备结束 Episode】
        └────────────────┬─────────────────┘
                         │ 否
                         ▼
        【写入 LeRobotDataset 标准帧】
```

### 8.1 规则细则：
1. **离合暂停不计入数据**：操作员按下 `SPACE` 暂停机械臂（为了重新调整人体手部姿态）期间，机器人保持不动。该阶段**绝对不可写入数据**，否则策略模型会学到“原地发呆”的模式崩溃。
2. **大幅重置切分 Episode**：当操作员按下 `R`（返回 READY 姿态）或 `H`（返回 HOME 姿态）准备下一轮推拿时，系统必须自动触发 `end_episode()` 并开启全新的 `start_episode()`。

---

## 9. 下游 VLA / Policy 训练与安全推理评估指南

### 9.1 模型超参数推荐配置 (SmolVLA-450M / ACT)

```yaml
# configs/train_tuina_22dof.yaml
fps: 30
model:
  type: smolvla # or act / diffusion
  chunk_size: 30 # 预测未来 1 秒动作序列 (30 步 action chunk)
  dim_model: 512
  n_action_steps: 15 # 每次滚动执行前 15 步 (Action Chunk Smoothing)

dataset:
  repo_id: "tuina_massage_22dof"
  observation_features:
    - "observation.state"
    - "observation.images.overhead_cam"
  action_feature: "action"
  normalization_mapping:
    "observation.state": "MEAN_STD"
    "action": "MEAN_STD"
```

### 9.2 推理部署强制安全包络拦截器 (Safety Envelope Interceptor)

下游模型推理在向底层硬件发送 `action` 前，**必须经过安全拦截层**：
1. **关节极限硬限幅**：所有 22 个维度的命令必须用 `np.clip(action, min_limits, max_limits)` 硬截断；
2. **单步跳变限速**：$|\Delta q_i| = |action_i - current\_state_i| \le \text{max\_dq}$（如机械臂单步 33ms 不超过 1.0°，防止神经网络输出离群值导致飞车）；
3. **急停热键绑定**：推理主循环中必须保留全局急停按键（如按下 `ESC` 或 `Q` 立即广播 CAN `0x0000 0xFE` 断电刹车）。

---

## 10. 接口对接常见问题 (FAQ) 与联调排错

### Q1: 灵巧手与机械臂的控制循环时间抖动如何处理？
**解答**：在 `Co_Teleop` 中，灵巧手采用 4Mbps 高速串口通信，机械臂采用 500kbps CAN 总线。在 30Hz（33.3ms）控制周期下，机械臂 6 轴广播仅耗时约 2ms，灵巧手 16 轴全读写约 5ms，总延迟在 8ms 内，留有充足裕度。务必使用 `time.perf_counter()` 配合高精度休眠。

### Q2: 为什么不直接在动作空间记录笛卡尔末端坐标 $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$？
**解答**：中医推拿涉及大量的肘部、大臂协同发力与揉捏角度。直接学习 22-DOF 关节角度不仅包含末端空间信息，还锁定了整条手臂的构型（避免自解算 IK 时的多解突变与奇异点自撞）。

### Q3: 穴位检测模型与 VLA 模型的先后调用关系是什么？
**解答**：
- **阶段一（宏观到达）**：穴位检测模型识别背部特定穴位（如大椎穴），逆解出机械臂初始预备姿态并平滑运动至穴位上方 3cm；
- **阶段二（微观推拿）**：触发 VLA 策略模型接管，依据背部视觉反馈与当前 22-DOF 姿态，执行揉法、滚法等动作。
