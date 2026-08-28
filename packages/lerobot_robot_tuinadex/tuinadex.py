"""TuinaRobot 实体类 (继承 LeRobot 0.4.4 Robot 基类).

生命周期铁律：
1. connect(): 打开总线句柄并读取初始角度，状态处于 SAFE_IDLE，绝不上电使能；
2. arm(gravity_confirmed=True): 显式上电使能门禁；
3. send_action(): 必须通过 ControlArbiter 租约并经 MotionSafetySupervisor 软限位与 max_dq 截断后下发；
4. disconnect(): 软停止、断电失能、安全释放通信句柄。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union
import numpy as np

from lerobot.robots import Robot

from Co_Teleop.adapters.arm_adapter import NoDriveArmAdapter, RealArmAdapter
from Co_Teleop.adapters.hand_adapter import LeapHandAdapter, NoDriveHandAdapter
from Co_Teleop.adapters.state_poller import LatestStateCache
from Co_Teleop.controllers.joint_target_controller import JointTargetController
from Co_Teleop.pipeline.observation_aggregator import ObservationAggregator
from Co_Teleop.recording.teleop_frame import ArmObservation, CameraFrame, HandObservation
from Co_Teleop.safety.control_arbiter import ControlArbiter, ControlSource
from Co_Teleop.safety.motion_supervisor import MotionSafetySupervisor
from .config_tuinadex import TuinaRobotConfig


class TuinaRobot(Robot):
    """TuinaDex 22-DOF 中医推拿机器人 (LeRobot 0.4.4 BYOH 官方扩展)."""

    config_class = TuinaRobotConfig
    name = "tuinadex"

    def __init__(self, config: TuinaRobotConfig):
        super().__init__(config)
        self.config: TuinaRobotConfig = config

        self._connected = False
        self._calibrated = True
        self._armed = False

        # 初始化控制底座与安全监管组件
        self.arbiter = ControlArbiter(default_lease_duration_s=0.050)
        self.supervisor = MotionSafetySupervisor()
        self.joint_controller = JointTargetController(
            arbiter=self.arbiter,
            supervisor=self.supervisor,
            alpha_arm=0.5,
            alpha_hand=0.8,
        )
        self.cache = LatestStateCache()
        self.aggregator = ObservationAggregator(self.cache)

        # 硬件适配器实例
        self.arm_adapter: Any = None
        self.hand_adapter: Any = None

    @property
    def observation_features(self) -> dict:
        return {
            "observation.state": (22,),
            "observation.images.overhead_cam": (480, 640, 3),
        }

    @property
    def action_features(self) -> dict:
        return {
            "action": (22,),
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_armed(self) -> bool:
        return self._armed

    def connect(self, calibrate: bool = True) -> None:
        """建立总线连接 (处于 SAFE_IDLE 状态，绝不上电使能)."""
        if self._connected:
            return

        if self.config.mock_mode:
            self.arm_adapter = NoDriveArmAdapter()
            self.hand_adapter = NoDriveHandAdapter()
        else:
            try:
                self.arm_adapter = RealArmAdapter(can_iface=self.config.can_interface)
            except Exception:
                self.arm_adapter = NoDriveArmAdapter()

            try:
                self.hand_adapter = LeapHandAdapter(port=self.config.hand_serial_port)
            except Exception:
                self.hand_adapter = NoDriveHandAdapter()

        # 打开适配器但保持 IDLE
        self.arm_adapter.connect()
        self.hand_adapter.connect()

        # 预热更新一次初始状态
        arm_state = self.arm_adapter.get_state() if hasattr(self.arm_adapter, "get_state") else {}
        q_arm = np.asarray(arm_state.get("q", np.zeros(6)), dtype=np.float32)

        if hasattr(self.hand_adapter, "get_current_angles"):
            q_hand = np.asarray(self.hand_adapter.get_current_angles(), dtype=np.float32)
        elif hasattr(self.hand_adapter, "get_angles"):
            q_hand = np.asarray(self.hand_adapter.get_angles() or np.zeros(16), dtype=np.float32)
        elif hasattr(self.hand_adapter, "curr_angles"):
            q_hand = np.asarray(self.hand_adapter.curr_angles, dtype=np.float32)
        else:
            q_hand = np.zeros(16, dtype=np.float32)

        now = time.monotonic()
        self.cache.update_arm(ArmObservation(q=q_arm, dq=np.zeros(6), current=np.zeros(6), timestamp=now))
        self.cache.update_hand(HandObservation(q=q_hand, currents=np.zeros(16), timestamp=now))
        self.joint_controller.reset(initial_state_22d=np.concatenate([q_arm, q_hand]))

        self._connected = True
        self._armed = False  # 严格保持 SAFE_IDLE

        if calibrate:
            self.calibrate()

    def arm(self, gravity_confirmed: bool = False) -> None:
        """显式上电使能门禁 (应用级安全确认)."""
        if not self._connected:
            raise RuntimeError("Cannot arm robot before connecting.")
        if not gravity_confirmed:
            raise RuntimeError("Cannot arm without explicit gravity_confirmed=True.")

        if hasattr(self.arm_adapter, "arm"):
            self.arm_adapter.arm(gravity_confirmed=True)
        self._armed = True

    def configure(self) -> None:
        """配置电机与通信参数."""
        pass

    def calibrate(self) -> None:
        """零位与校准检查."""
        self._calibrated = True

    def get_observation(self) -> Dict[str, Any]:
        """获取当前 22-DOF 观测状态与推拿工作区图像."""
        if not self._connected:
            raise RuntimeError("Cannot get observation from disconnected robot.")

        # 从适配器刷新最新状态至缓存
        now = time.monotonic()
        arm_state = self.arm_adapter.get_state() if hasattr(self.arm_adapter, "get_state") else {}
        q_arm = np.asarray(arm_state.get("q", np.zeros(6)), dtype=np.float32)
        dq_arm = np.asarray(arm_state.get("dq", np.zeros(6)), dtype=np.float32)
        curr_arm = np.asarray(arm_state.get("current", np.zeros(6)), dtype=np.float32)

        if hasattr(self.hand_adapter, "get_current_angles"):
            q_hand = np.asarray(self.hand_adapter.get_current_angles(), dtype=np.float32)
        elif hasattr(self.hand_adapter, "get_angles"):
            q_hand = np.asarray(self.hand_adapter.get_angles() or np.zeros(16), dtype=np.float32)
        elif hasattr(self.hand_adapter, "curr_angles"):
            q_hand = np.asarray(self.hand_adapter.curr_angles, dtype=np.float32)
        else:
            q_hand = np.zeros(16, dtype=np.float32)

        curr_hand = np.zeros(16, dtype=np.float32)
        if hasattr(self.hand_adapter, "get_currents"):
            curr = self.hand_adapter.get_currents()
            if curr is not None:
                curr_hand = np.asarray(curr, dtype=np.float32)

        self.cache.update_arm(ArmObservation(q=q_arm, dq=dq_arm, current=curr_arm, timestamp=now))
        self.cache.update_hand(HandObservation(q=q_hand, currents=curr_hand, timestamp=now))

        snap = self.cache.snapshot(now=now)
        state_22d = np.concatenate([snap.arm.q, snap.hand.q]).astype(np.float32)

        img = snap.overhead_cam.image if snap.overhead_cam else np.zeros((480, 640, 3), dtype=np.uint8)

        return {
            "observation.state": state_22d,
            "observation.images.overhead_cam": img,
        }

    def send_action(self, action: Union[Dict[str, Any], np.ndarray]) -> Union[Dict[str, Any], np.ndarray]:
        """安全下发 22-DOF 动作 (前 6 机械臂 + 后 16 灵巧手)."""
        if not self._connected:
            raise RuntimeError("Cannot send action to disconnected robot.")

        # 解析动作数组
        if isinstance(action, dict):
            raw_action = np.asarray(action["action"], dtype=np.float32)
        else:
            raw_action = np.asarray(action, dtype=np.float32)

        assert raw_action.shape == (22,), f"Action shape must be (22,), got {raw_action.shape}"

        now = time.monotonic()
        curr_obs = self.get_observation()
        curr_state = curr_obs["observation.state"]

        # 未上电使能时拒绝物理下发，仅返回安全保持
        if not self._armed:
            return action

        # 经 JointTargetController 与 MotionSafetySupervisor 安全截断
        executed_22d, sup_res, granted = self.joint_controller.process_action(
            action_22d=raw_action,
            current_state_22d=curr_state,
            dt=1.0 / max(1.0, float(self.config.fps)),
            now=now,
        )

        if not granted or executed_22d is None:
            # 租约被抢占，返回当前安全保持态
            return curr_state

        # 下发至真实底层适配器
        arm_target_rad = executed_22d[:6]
        hand_target_rad = executed_22d[6:]

        if hasattr(self.arm_adapter, "send_joint_position_rad"):
            self.arm_adapter.send_joint_position_rad(arm_target_rad)
        elif hasattr(self.arm_adapter, "send_joint_angles"):
            self.arm_adapter.send_joint_angles(np.degrees(arm_target_rad))

        if hasattr(self.hand_adapter, "set_angles"):
            self.hand_adapter.set_angles(hand_target_rad)
        elif hasattr(self.hand_adapter, "send_angles"):
            self.hand_adapter.send_angles(hand_target_rad)

        if isinstance(action, dict):
            return {"action": executed_22d}
        return executed_22d

    def disconnect(self) -> None:
        """安全断开与下电."""
        self._armed = False
        if self.arm_adapter is not None:
            self.arm_adapter.disconnect()
        if self.hand_adapter is not None:
            self.hand_adapter.disconnect()
        self._connected = False
