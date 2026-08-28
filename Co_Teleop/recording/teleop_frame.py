"""TeleopFrame 强类型数据契约 (Data Contract).

统一时空对齐快照：
- 包含主时钟 timestamp 与数据关联索引 frame_id
- 汇聚机械臂 6-DOF、灵巧手 16-DOF 真实观测与目标动作
- 整合推拿工作区工业相机图像与 AcuPointDet 37 个穴位检测结构化特征
- 提供状态有效性标记 (FrameValidity) 与任务元数据 (TaskMetadata)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional

import numpy as np


class FrameValidity(str, Enum):
    """单帧数据有效性枚举."""

    VALID = "VALID"                       # 正常有效示范帧，可直接用于训练
    PAUSED = "PAUSED"                     # 空格离合暂停状态 (不计入训练集)
    HAND_LOCKED = "HAND_LOCKED"           # 灵巧手处于 L 键姿态锁定状态
    VISION_DEGRADED = "VISION_DEGRADED"   # 视觉追踪置信度下降/部分遮挡
    VISION_LOST = "VISION_LOST"           # 视觉丢失 (看门狗介入)
    SAFETY_STOP = "SAFETY_STOP"           # 触发限速/限幅阻尼减速
    ESTOP = "ESTOP"                       # 触发急停


@dataclass
class CameraFrame:
    """单个相机的图像帧与采集时间戳."""

    image: np.ndarray                    # RGB 图像 (H, W, 3) uint8
    timestamp: float                     # 硬件/驱动取帧单调时间 (秒)
    frame_id: int = 0                    # 相机局部帧序号
    camera_id: str = "overhead_cam"      # 相机标识 (overhead_cam, operator_cam, wrist_cam)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "shape": list(self.image.shape) if self.image is not None else None,
        }


@dataclass
class ArmObservation:
    """机械臂 6-DOF 实时观测状态."""

    q: np.ndarray                        # 6 轴当前关节角度 (单位: rad, 形状 (6,))
    dq: np.ndarray                       # 6 轴当前关节角速度 (单位: rad/s, 形状 (6,))
    current: np.ndarray                  # 6 轴当前相电流/力矩表征 (形状 (6,))
    flags: int = 0                       # 电机状态标志位 (如到位、使能状态)
    timestamp: float = 0.0               # 读取到底层 CAN 状态的单调时间 (秒)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=np.float32)
        self.dq = np.asarray(self.dq, dtype=np.float32)
        self.current = np.asarray(self.current, dtype=np.float32)


@dataclass
class HandObservation:
    """LEAP Hand 16-DOF 灵巧手实时观测状态."""

    q: np.ndarray                        # 16 轴物理弯曲弧度 (单位: rad, 形状 (16,))
    currents: np.ndarray                 # 16 轴实时电流/负载 (形状 (16,))
    timestamp: float = 0.0               # 串口读取时间 (秒)

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=np.float32)
        self.currents = np.asarray(self.currents, dtype=np.float32)


@dataclass
class AcupointObservation:
    """AcuPointDet 背部推拿穴位检测结果."""

    keypoints_2d: np.ndarray             # 37 个穴位 2D 坐标与置信度 (形状 (37, 3), [x, y, score])
    bbox: np.ndarray                     # 人体背部检测框 [x1, y1, x2, y2] (形状 (4,))
    confidence: float = 0.0              # 总体检测置信度
    timestamp: float = 0.0               # 推理完成单调时间 (秒)

    def __post_init__(self):
        self.keypoints_2d = np.asarray(self.keypoints_2d, dtype=np.float32)
        self.bbox = np.asarray(self.bbox, dtype=np.float32)


@dataclass
class TaskMetadata:
    """当前采集任务的语义标签."""

    task_name: str = "中医推拿示范"       # 任务中文名称 (如 "按揉大椎穴 30 次")
    technique: str = "kneading"          # 手法代码 (kneading, rolling, point_press, stroking)
    phase: str = "kneading"              # 阶段 (approach, contact, kneading, lift, finish)
    target_acupoint: str = "dazhui"      # 目标穴位拼音标识 (dazhui, jianjing, feishu 等)
    episode_id: str = ""                 # 当前 Episode 唯一标识
    step_idx: int = 0                    # Episode 内的步数索引


@dataclass
class TeleopFrame:
    """统一时空对齐快照 (全模态单帧数据契约)."""

    frame_id: int                        # 全局帧序号
    timestamp: float                     # 主时钟时间戳 (系统单调时钟 time.monotonic())

    # 1. 底层本体观测
    arm: ArmObservation
    hand: HandObservation
    observation_state_22d: np.ndarray    # 22-DOF 关节弧度拼合 (前 6 机械臂, 后 16 灵巧手)

    # 2. 动作空间
    action_22d: np.ndarray               # 控制器最终执行目标 22-DOF 关节弧度 (rad)
    cartesian_cmd: np.ndarray            # 人类遥操原始意图 6-DOF 笛卡尔速度 [vx, vy, vz, wx, wy, wz]

    # 3. 视觉与环境感知
    overhead_cam: Optional[CameraFrame] = None
    acupoints: Optional[AcupointObservation] = None

    # 4. 状态与语义
    validity: FrameValidity = FrameValidity.VALID
    task: TaskMetadata = field(default_factory=TaskMetadata)

    def __post_init__(self):
        self.observation_state_22d = np.asarray(self.observation_state_22d, dtype=np.float32)
        self.action_22d = np.asarray(self.action_22d, dtype=np.float32)
        self.cartesian_cmd = np.asarray(self.cartesian_cmd, dtype=np.float32)

        assert self.observation_state_22d.shape == (22,), f"observation_state_22d shape must be (22,), got {self.observation_state_22d.shape}"
        assert self.action_22d.shape == (22,), f"action_22d shape must be (22,), got {self.action_22d.shape}"
        assert self.cartesian_cmd.shape == (6,), f"cartesian_cmd shape must be (6,), got {self.cartesian_cmd.shape}"

    def to_lerobot_dict(self) -> Dict[str, Any]:
        """转换为 LeRobotDataset.add_frame() 消费的标准字典格式."""
        item: Dict[str, Any] = {
            "observation.state": self.observation_state_22d,
            "action": self.action_22d,
            "task": self.task.task_name,
        }

        if self.overhead_cam is not None and self.overhead_cam.image is not None:
            item["observation.images.overhead_cam"] = self.overhead_cam.image

        if self.acupoints is not None and self.acupoints.keypoints_2d is not None:
            item["observation.environment.acupoints"] = self.acupoints.keypoints_2d

        return item

    def to_raw_dict(self) -> Dict[str, Any]:
        """转换为 RawRecorder 保存的完整遥测字典 (含内部控制与诊断变量)."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "arm": {
                "q": self.arm.q.tolist(),
                "dq": self.arm.dq.tolist(),
                "current": self.arm.current.tolist(),
                "flags": int(self.arm.flags),
                "timestamp": self.arm.timestamp,
            },
            "hand": {
                "q": self.hand.q.tolist(),
                "currents": self.hand.currents.tolist(),
                "timestamp": self.hand.timestamp,
            },
            "observation_state_22d": self.observation_state_22d.tolist(),
            "action_22d": self.action_22d.tolist(),
            "cartesian_cmd": self.cartesian_cmd.tolist(),
            "overhead_cam": self.overhead_cam.to_dict() if self.overhead_cam else None,
            "acupoints": {
                "keypoints_2d": self.acupoints.keypoints_2d.tolist(),
                "bbox": self.acupoints.bbox.tolist(),
                "confidence": float(self.acupoints.confidence),
                "timestamp": self.acupoints.timestamp,
            } if self.acupoints else None,
            "validity": self.validity.value,
            "task": {
                "task_name": self.task.task_name,
                "technique": self.task.technique,
                "phase": self.task.phase,
                "target_acupoint": self.task.target_acupoint,
                "episode_id": self.task.episode_id,
                "step_idx": self.task.step_idx,
            },
        }
