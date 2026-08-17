"""
MassageRobotV2 — 22-DOF 机械臂+灵巧手合并 Robot 子类 (占位骨架).

完整设计见 Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md §3.3.
本文档仅提供顶层占位 + 接口约定, 不含实现细节 (待 Phase A 启动后实现).

设计要点:
  - 观察空间: 22 维 (6 臂 + 16 手) + 相机
  - 动作空间: 22 维 (与观察同, LeRobot 习惯延迟状态作为 BC label)
  - 双总线: STM32 串口 (臂) + Dynamixel 4M (手), MassageRobotV2 内部管 bus
  - 安全: 急停一键停两个子系统
"""
from __future__ import annotations


class MassageRobotV2Config:  # 占位, 真实实现继承 RobotConfig
    """22-DOF MassageRobot config (待 Phase A 实现)."""
    # 臂 (6-DOF)
    arm_port: str = "/dev/ttyUSB0"
    arm_baudrate: int = 115200
    arm_transport: str = "can"  # "serial" / "can" (ADR-005)
    # 手 (16-DOF)
    hand_port: str = "/dev/ttyUSB1"
    hand_baudrate: int = 4_000_000
    hand_motor_ids: list[int] = list(range(16))
    # 相机
    cameras: dict = {}


class MassageRobotV2:  # 占位, 真实实现继承 Robot
    """22-DOF 机械臂+灵巧手 LeRobot Robot (待 Phase A 实现)."""
    config_class = MassageRobotV2Config

    def __init__(self, config: MassageRobotV2Config):
        self.config = config
        self._arm = None  # lerobot_robot_massage.MassageRobot
        self._hand = None  # leap_hand_utils.dynamixel_client.DynamixelClient

    def connect(self) -> None:
        raise NotImplementedError("Phase A 待实现")

    def disconnect(self) -> None:
        raise NotImplementedError("Phase A 待实现")

    def get_observation(self) -> dict:
        """22 维 state + 相机帧."""
        raise NotImplementedError("Phase A 待实现")

    def send_action(self, action: dict) -> dict:
        """按 key 分发: 6 臂 → set_joints, 16 手 → set_positions."""
        raise NotImplementedError("Phase A 待实现")

    def emergency_stop(self) -> None:
        """一键急停两个子系统 (臂 + 手)."""
        raise NotImplementedError("Phase A 待实现")