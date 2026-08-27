"""scripts/teleop/arm_adapter.py — 仿真/真机臂统一接口 (spec TASK-02/24).

SimulationArmAdapter: 复用 ArmClient (MuJoCo socket), 保持仿真遥操能力.
RealArmAdapter: 封装 CartesianController → ZdtController, 是唯一真机笛卡尔入口;
  不实现 CAN 协议 / 不重复 IK / 不直接操作电机帧. 视觉层只产 CartesianCommand.
"""
from __future__ import annotations

import numpy as np

from lerobot_robot_massage.zdt.types import (
    CartesianCommand, EEPose, JointState,
)


class SimulationArmAdapter:
    """MuJoCo 仿真臂 (ArmClient socket 协议)."""

    def __init__(self, arm_client):
        self._arm = arm_client

    def connect(self) -> None:
        self._arm.remote_enable()

    def disconnect(self) -> None:
        self._arm.remote_disable()
        self._arm.close()

    def get_joint_state(self) -> JointState:
        angles, vels, loads = self._arm.get_state()
        return JointState(q=tuple(angles), dq=tuple(vels),
                          current_ma=tuple(loads))

    def get_ee_pose(self) -> EEPose:
        ep = self._arm.get_ee_pose()
        if ep is None:
            raise RuntimeError("仿真未返回末端位姿 (get_ee_pose)")
        pos_m, quat = ep
        position = np.asarray(pos_m, float) * 1000.0          # m → mm
        rotation = _quat_to_rotmat(quat)
        return EEPose(position=position, rotation=rotation)

    def move_cartesian_velocity(self, cmd: CartesianCommand) -> None:
        # 仿真协议约定: end_event 输入按 mm/s + rad/s 解释 (回归测试同约定)
        self._arm.end_event(*cmd.twist)

    def ready(self) -> None:
        self._arm.soft_reset()

    def home(self) -> None:
        self._arm.soft_reset()

    def reset(self) -> None:
        self.ready()

    def e_stop(self) -> None:
        self._arm.e_stop()

    def state(self) -> str:
        return "TELEOP"


class NoDriveArmAdapter:
    """空跑/只显示适配器 (no-drive): 未连接物理臂/CAN时用于视觉遥操与UI测试."""

    def __init__(self, ready_pose: Optional[list] = None, home_pose: Optional[list] = None,
                 joint_limits: Optional[list] = None):
        self._phase = "NO_DRIVE"
        self.ready_pose = ready_pose or [0.0, 75.0, 55.0, 0.0, 130.0, 0.0]
        self.home_pose = home_pose or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_limits = joint_limits
        self._q = list(self.home_pose)

    def connect(self) -> None:
        self._phase = "SAFE_IDLE"

    def arm(self, gravity_confirmed: bool = False) -> None:
        self._phase = "ARMED"

    def enter_teleop(self) -> None:
        self._phase = "TELEOP"

    def exit_teleop(self) -> None:
        self._phase = "ARMED"

    def disconnect(self) -> None:
        self._phase = "DISCONNECTED"

    def get_joint_state(self) -> JointState:
        return JointState(q=tuple(self._q), dq=(0.0,) * 6, current_ma=(0.0,) * 6,
                          flags=(0,) * 6, status=self._phase)

    def get_real_joint_angles(self) -> list[float]:
        return list(self._q)

    def get_ee_pose(self) -> EEPose:
        return EEPose(position=np.zeros(3), rotation=np.eye(3))

    def move_cartesian_velocity(self, cmd: CartesianCommand) -> None:
        pass

    def step_pose(self, p_des, R_des, **kw) -> None:
        pass

    def re_arm(self, gravity_confirmed: bool = True) -> None:
        if self._phase == "STOPPED":
            self._phase = "TELEOP"

    def ready(self) -> None:
        self._phase = "TELEOP"
        self._q = list(self.ready_pose)

    def home(self) -> None:
        self._phase = "TELEOP"
        self._q = list(self.home_pose)

    def reset(self) -> None:
        self.ready()

    def e_stop(self) -> None:
        self._phase = "STOPPED"

    def state(self) -> str:
        return self._phase


class RealArmAdapter:
    """真机臂: 封装 CartesianController (spec §6.1). 不直接操作 CAN/IK.

    P0-①: connect() 只到 SAFE_IDLE (枚举+验证), 不自动 arm/使能扭矩;
    arm(gravity_confirmed) 由调用方显式调用 (重力关节 J2/J3 需确认).
    """

    def __init__(self, ctrl, ready_pose: Optional[list] = None, home_pose: Optional[list] = None,
                 joint_limits: Optional[list] = None, joint_limit_margin_deg: Optional[float] = None,
                 use_tracked: bool = True,
                 **cart_kwargs):
        from lerobot_robot_massage.zdt.cartesian import CartesianController
        self._ctrl = ctrl
        if joint_limits is not None:
            self._ctrl.config.limits = [tuple(lim) for lim in joint_limits]
            cart_kwargs["joint_limits"] = [tuple(lim) for lim in joint_limits]
        if joint_limit_margin_deg is not None:
            self._ctrl.config.joint_limit_margin_deg = float(joint_limit_margin_deg)
            cart_kwargs["joint_limit_margin_deg"] = float(joint_limit_margin_deg)
        cart_kwargs["use_tracked"] = use_tracked
        self._cart = CartesianController(ctrl, **cart_kwargs)
        self.ready_pose = ready_pose
        self.home_pose = home_pose
        self.joint_limits = joint_limits

    def connect(self) -> None:
        self._ctrl.connect()                       # SAFE_IDLE, 不使能扭矩

    def arm(self, gravity_confirmed: bool = False) -> None:
        """显式臂置 (使能扭矩) — 调用方在用户确认后调用."""
        self._ctrl.arm(gravity_confirmed)

    def enter_teleop(self) -> None:
        self._ctrl.enter_teleop()

    def exit_teleop(self) -> None:
        self._ctrl.exit_teleop()

    def re_arm(self, gravity_confirmed: bool = True) -> None:
        """从 STOPPED / FAULT 状态安全恢复至 ARMED / TELEOP 状态."""
        try:
            if self._ctrl.robot.phase.name in ("STOPPED", "FAULT"):
                self._ctrl.robot.re_arm(confirmed=True)
                self._ctrl.arm(gravity_confirmed=gravity_confirmed)
                self._ctrl.enter_teleop()
        except Exception:
            pass

    def disconnect(self) -> None:
        try:
            if self._ctrl.robot.phase.name in ("ARMED", "TELEOP"):
                self._ctrl.disarm()
        except Exception:
            pass
        finally:
            try:
                self._ctrl.disconnect()
            except Exception:
                pass

    def get_joint_state(self) -> JointState:
        if hasattr(self._cart, "_q_tracked") and self._cart._q_tracked is not None:
            q = list(self._cart._q_tracked)
        elif hasattr(self._ctrl, "_tracked_angles") and self._ctrl._tracked_angles is not None:
            q = list(self._ctrl._tracked_angles)
        else:
            q = [0.0] * 6
        return JointState(q=tuple(q), dq=(0.0,) * 6,
                          current_ma=(0.0,) * 6,
                          flags=(0,) * 6,
                          status=self._ctrl.robot.phase.name)

    def get_real_joint_angles(self) -> list[float]:
        return self._ctrl.read_real_angles(use_kb=True)

    def get_ee_pose(self) -> EEPose:
        # P1-⑥: 经公共接口 get_current_pose(), 不依赖 Controller 私有 FK
        return self._cart.get_current_pose()

    def move_cartesian_velocity(self, cmd: CartesianCommand) -> None:
        self._cart.step(*cmd.linear_velocity, *cmd.angular_velocity,
                        cmd_ts=cmd.timestamp)

    def step_pose(self, p_des, R_des, **kw) -> None:
        self._cart.step_pose(p_des, R_des, **kw)

    def ready(self) -> None:
        """安全运动至按摩准备姿态 (配置 target_pose 或 READY_POSE_DEG), 100 RPM 同步."""
        self.re_arm(gravity_confirmed=True)
        self._cart.ready(target_angles_deg=self.ready_pose)

    def home(self) -> None:
        """安全运动回上电初始姿态 (配置 target_pose 或 JOINT_INIT_ANGLE_DEG), 100 RPM 同步."""
        self.re_arm(gravity_confirmed=True)
        self._cart.home(target_angles_deg=self.home_pose)

    def reset(self) -> None:
        self.ready()

    def e_stop(self) -> None:
        self._ctrl.e_stop()

    def state(self) -> str:
        return self._ctrl.robot.phase.name


def _quat_to_rotmat(q) -> np.ndarray:
    """(w,x,y,z) → SO(3) (测试/仿真用)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


SimArmAdapter = SimulationArmAdapter
ArmAdapter = RealArmAdapter

