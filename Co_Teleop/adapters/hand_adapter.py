"""scripts/teleop/hand_adapter.py — 灵巧手硬件抽象适配层 (LEAP Hand Hardware Adapter).

【设计规范】
1. 提供统一的灵巧手控制接口 (LeapHandAdapter 与 NoDriveHandAdapter)；
2. 遵循实体硬件安全规范：
   - 自动加载 motor_limits.json 机械物理限位表并在写入舵机前进行严密裁剪；
   - 保证未连接/断开时的安全状态管理；
   - 提供平滑回全开位 (Relax to OPEN) 丢帧保护；
3. 支持纯仿真与单元测试的 NoDrive 模式。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# 关节驱动方向符号映射表 (针对 LEAP Hand 右手 16 舵机)
JOINT_DIR = np.array([-1, -1, -1, -1,
                      -1, -1, -1, -1,
                      -1, -1, -1, -1,
                       1, -1, -1, -1], dtype=np.float64)

# 默认全开位姿 (rad)
DEFAULT_OPEN_POSE = np.array([
    3.1155, 4.6204, 3.2076, 1.5785,
    3.2413, 3.0618, 3.1170, 4.5927,
    3.1186, 3.0756, 3.1799, 4.6004,
    3.1186, 1.5555, 4.6234, 4.7247,
], dtype=np.float64)


class NoDriveHandAdapter:
    """灵巧手空跑适配器 (用于无实体硬件或离线测试)."""

    def __init__(self, open_pose: Optional[np.ndarray] = None):
        self.open_pose = open_pose.copy() if open_pose is not None else DEFAULT_OPEN_POSE.copy()
        self.curr_pos = self.open_pose.copy()
        self.curr_angles = np.zeros(16, dtype=np.float64)
        self._connected = False
        self._phase = "UNPOWERED"
        self.loss_t0: Optional[float] = None
        self.relax_from: Optional[np.ndarray] = None

    def connect(self, port: Optional[str] = None) -> bool:
        self._connected = True
        self._phase = "POWERED_OPEN"
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._phase = "DISCONNECTED"

    def is_connected(self) -> bool:
        return self._connected

    def set_angles(self, angles_16: np.ndarray) -> np.ndarray:
        """输入 16 维相对全开位关节弯曲角度 (rad) -> 更新模拟电机位置."""
        self.curr_angles = np.asarray(angles_16, dtype=np.float64)
        self.curr_pos = self.open_pose + JOINT_DIR * self.curr_angles
        self._phase = "TELEOP"
        return self.curr_pos

    def set_open(self) -> np.ndarray:
        """复位回全开位."""
        self.curr_angles = np.zeros(16, dtype=np.float64)
        self.curr_pos = self.open_pose.copy()
        self._phase = "OPEN"
        return self.curr_pos

    def relax_step(self, now: float, hold_time: float = 0.30, ramp_time: float = 0.60) -> np.ndarray:
        """空跑模式下的平滑回全开位 (Relax to OPEN)."""
        if self.loss_t0 is None:
            self.loss_t0 = now
            self.relax_from = self.curr_pos.copy()

        elapsed = now - self.loss_t0
        if elapsed < hold_time:
            self.curr_pos = self.relax_from.copy()
        else:
            t = min(1.0, (elapsed - hold_time) / max(0.01, ramp_time))
            self.curr_pos = self.relax_from + (self.open_pose - self.relax_from) * t
            self.curr_angles = (self.curr_pos - self.open_pose) * JOINT_DIR

        self._phase = "RELAX"
        return self.curr_pos

    def reset_loss_state(self) -> None:
        """重置手部丢失回退计时器."""
        self.loss_t0 = None
        self.relax_from = None

    def get_positions(self) -> np.ndarray:
        return self.curr_pos.copy()

    def get_current_angles(self) -> np.ndarray:
        return self.curr_angles.copy()

    def state(self) -> str:
        return self._phase


class LeapHandAdapter:
    """LEAP Hand 16-DOF 实体硬件适配器."""

    def __init__(self, port: Optional[str] = None,
                 kP: int = 300, kI: int = 0, kD: int = 100, curr_lim: int = 150,
                 limits_path: Optional[str | Path] = None):
        self.port = port
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.curr_lim = curr_lim
        self.leap = None
        self.open_pose = DEFAULT_OPEN_POSE.copy()
        self.motor_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_commanded_pose: Optional[np.ndarray] = None
        self.loss_t0: Optional[float] = None
        self.relax_from: Optional[np.ndarray] = None
        self.curr_angles = np.zeros(16, dtype=np.float64)

        # 尝试载入 OPEN_POSE 与 motor_limits
        self._load_config_and_limits(limits_path)

    def _load_config_and_limits(self, limits_path: Optional[str | Path]) -> None:
        """从 Leap_Hand 模块加载真实全开位和电机物理限位."""
        # 1. 尝试导入 Leap_Hand.python.main 中的 OPEN_POSE
        try:
            import sys
            leap_pkg = Path(__file__).resolve().parents[3] / "Leap_Hand" / "python"
            if str(leap_pkg) not in sys.path:
                sys.path.insert(0, str(leap_pkg))
            from main import OPEN_POSE as REAL_OPEN_POSE
            self.open_pose = REAL_OPEN_POSE.copy()
        except Exception:
            self.open_pose = DEFAULT_OPEN_POSE.copy()

        # 2. 尝试载入 motor_limits.json
        if limits_path is None:
            limits_path = Path(__file__).resolve().parents[3] / "Leap_Hand" / "python" / "gesture_mapping" / "motor_limits.json"
        else:
            limits_path = Path(limits_path)

        if limits_path.exists():
            try:
                with open(limits_path, "r", encoding="utf-8") as f:
                    lim = json.load(f)
                self.motor_limits = (
                    np.array(lim["min"], dtype=np.float64),
                    np.array(lim["max"], dtype=np.float64),
                )
            except Exception as e:
                print(f"[灵巧手警告] 加载限位表失败 ({e})，不应用硬限位裁剪")

    def connect(self, port: Optional[str] = None) -> bool:
        """延迟上电连接 LEAP Hand 串口."""
        if self.leap is not None:
            return True
        target_port = port or self.port
        try:
            import sys
            leap_pkg = Path(__file__).resolve().parents[3] / "Leap_Hand" / "python"
            if str(leap_pkg) not in sys.path:
                sys.path.insert(0, str(leap_pkg))
            from main import LeapNode

            self.leap = LeapNode(port=target_port, kP=self.kP, kI=self.kI, kD=self.kD, curr_lim=self.curr_lim)
            self.last_commanded_pose = self.open_pose.copy()
            print(f"[灵巧手] 成功连接并上电 (全开位): port={target_port} (kP={self.kP}, kD={self.kD}, curr_lim={self.curr_lim}mA)")
            return True
        except Exception as e:
            print(f"[灵巧手错误] 连接失败: {e}")
            self.leap = None
            return False

    def disconnect(self) -> None:
        """安全释放并断开灵巧手连接."""
        if self.leap is not None:
            try:
                self.set_open()
                self.leap.disconnect()
            except Exception:
                pass
            finally:
                self.leap = None

    def is_connected(self) -> bool:
        return self.leap is not None

    def set_angles(self, angles_16: np.ndarray) -> np.ndarray:
        """输入 16 维相对弯曲角 (rad) -> 计算绝对目标并下发."""
        self.curr_angles = np.asarray(angles_16, dtype=np.float64)
        pose = self.open_pose + JOINT_DIR * self.curr_angles

        # 电机物理限位安全裁剪
        if self.motor_limits is not None:
            pose = np.clip(pose, self.motor_limits[0], self.motor_limits[1])

        if self.leap is not None:
            self.leap.set_leap(pose)
            self.last_commanded_pose = pose.copy()

        return pose

    def set_open(self) -> np.ndarray:
        """复位至全开安全位."""
        self.curr_angles = np.zeros(16, dtype=np.float64)
        pose = self.open_pose.copy()
        if self.leap is not None:
            self.leap.set_open()
            self.last_commanded_pose = pose.copy()
        return pose

    def relax_step(self, now: float, hold_time: float = 0.30, ramp_time: float = 0.60) -> np.ndarray:
        """
        手部丢失时的平滑回全开位保护逻辑 (Relax to OPEN).
        - 丢失初期 (< hold_time): 保持最后一帧位置不晃动;
        - 持续丢失 (hold_time ~ hold_time + ramp_time): 线性平滑插值回到全开位;
        - 超过恢复期: 稳定维持在全开位.
        """
        if self.loss_t0 is None:
            self.loss_t0 = now
            self.relax_from = (self.last_commanded_pose.copy()
                               if self.last_commanded_pose is not None
                               else self.open_pose.copy())

        elapsed = now - self.loss_t0
        if elapsed < hold_time:
            pose = self.relax_from.copy()
        else:
            t = min(1.0, (elapsed - hold_time) / max(0.01, ramp_time))
            pose = self.relax_from + (self.open_pose - self.relax_from) * t

        if self.motor_limits is not None:
            pose = np.clip(pose, self.motor_limits[0], self.motor_limits[1])

        if self.leap is not None:
            self.leap.set_leap(pose)
            self.last_commanded_pose = pose.copy()

        return pose

    def reset_loss_state(self) -> None:
        """重置手部丢失回退计时器 (当检测到有效手势时调用)."""
        self.loss_t0 = None
        self.relax_from = None

    def get_positions(self) -> Optional[np.ndarray]:
        if self.leap is not None:
            try:
                return np.asarray(self.leap.read_pos(), dtype=np.float64)
            except Exception:
                pass
        return self.last_commanded_pose.copy() if self.last_commanded_pose is not None else self.open_pose.copy()

    def get_current_angles(self) -> np.ndarray:
        return self.curr_angles.copy()

    def state(self) -> str:
        if self.leap is None:
            return "UNPOWERED"
        return "POWERED"
