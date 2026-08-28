"""TuinaRobotConfig 配置类 (继承 LeRobot 0.4.4 RobotConfig)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.robots import RobotConfig


@dataclass
class TuinaRobotConfig(RobotConfig):
    """TuinaDex 22-DOF 中医推拿机器人配置."""

    can_interface: str = "can0"            # SocketCAN 接口名称
    can_bitrate: int = 500000              # CAN 波特率 (500kbps)
    hand_serial_port: str = "/dev/ttyUSB0" # 灵巧手 USB-TTL 串口号
    hand_baudrate: int = 4000000           # Dynamixel 4Mbps 波特率
    overhead_cam_id: str | int = 0         # 工作区工业相机设备标识
    fps: int = 30                          # 控制与采集标准帧率
    mock_mode: bool = False                # 是否启用 Mock 虚拟硬件模式
