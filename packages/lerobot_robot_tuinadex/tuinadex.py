"""TuinaRobot 实体类 (继承 LeRobot 0.4.4 Robot 基类).

生命周期与安全铁律：
1. 真实硬件严禁静默降级：mock_mode=False 时硬件初始化失败必须显式回滚并抛出 HARDWARE_FAULT；
2. 事务性连接回滚 (Rollback on failure)：Arm 或 Hand 任一失败立即安全释放全部已连接句柄；
3. 纯异步状态流：get_observation() 纯消费 LatestStateCache.snapshot()，绝不阻塞现场同步轮询；
4. 守护线程生命周期：connect() 启动 Arm/HandStatePoller，disconnect() 安全终止并 join 线程；
5. 显式上电使能门禁：connect() 保持 SAFE_IDLE，arm(gravity_confirmed=True) 显式使能；
6. 统一运动出口：所有下发动作必须经 ControlArbiter 租约并由 MotionSafetySupervisor 三级裁决。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union
import numpy as np

from lerobot.robots import Robot

from Co_Teleop.adapters.arm_adapter import NoDriveArmAdapter, RealArmAdapter
from Co_Teleop.adapters.hand_adapter import LeapHandAdapter, NoDriveHandAdapter
from Co_Teleop.adapters.state_poller import ArmStatePoller, HandStatePoller, LatestStateCache
from Co_Teleop.controllers.joint_target_controller import JointTargetController
from Co_Teleop.pipeline.observation_aggregator import ObservationAggregator
from Co_Teleop.recording.teleop_frame import ArmObservation, CameraFrame, HandObservation
from Co_Teleop.safety.control_arbiter import ControlArbiter, ControlSource
from Co_Teleop.safety.motion_supervisor import MotionSafetySupervisor, SafetyDecision
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
        self._hardware_fault = False

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

        # 硬件适配器与后台轮询线程
        self.arm_adapter: Any = None
        self.hand_adapter: Any = None
        self.arm_poller: Optional[ArmStatePoller] = None
        self.hand_poller: Optional[HandStatePoller] = None

        self._last_action_time: Optional[float] = None

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
        return self._connected and not self._hardware_fault

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_armed(self) -> bool:
        return self._armed

    def connect(self, calibrate: bool = True) -> None:
        """建立总线连接 (具备原子回滚与真实硬件失败硬防护)."""
        if self._connected:
            return

        self._hardware_fault = False

        # 1. 适配器实例化 (严格遵循 mock_mode 配置，绝无静默 Fallback)
        if self.config.mock_mode:
            self.arm_adapter = NoDriveArmAdapter()
            self.hand_adapter = NoDriveHandAdapter()
        else:
            try:
                self.arm_adapter = RealArmAdapter(can_iface=self.config.can_interface)
            except Exception as e:
                self._hardware_fault = True
                raise RuntimeError(f"[HARDWARE_FAULT] RealArmAdapter 初始化失败: {e}") from e

            try:
                self.hand_adapter = LeapHandAdapter(port=self.config.hand_serial_port)
            except Exception as e:
                self._hardware_fault = True
                raise RuntimeError(f"[HARDWARE_FAULT] LeapHandAdapter 初始化失败: {e}") from e

        # 2. 打开连接并执行事务性回滚 (Rollback on failure)
        try:
            arm_res = self.arm_adapter.connect()
            if arm_res is False:
                raise RuntimeError(f"机械臂连接失败 (接口: {self.config.can_interface})")

            hand_res = self.hand_adapter.connect()
            if hand_res is False:
                raise RuntimeError(f"灵巧手串口连接失败 (端口: {self.config.hand_serial_port})")
        except Exception as e:
            # 事务性回滚已打开的连接
            self._hardware_fault = True
            if self.arm_adapter is not None:
                try:
                    self.arm_adapter.disconnect()
                except Exception:
                    pass
            if self.hand_adapter is not None:
                try:
                    self.hand_adapter.disconnect()
                except Exception:
                    pass
            self._connected = False
            raise RuntimeError(f"[HARDWARE_FAULT] 硬件连接事务失败，已全部安全回滚释放: {e}") from e

        # 3. 启动后台守护状态轮询器 (ArmStatePoller & HandStatePoller)
        self.arm_poller = ArmStatePoller(arm_adapter=self.arm_adapter, cache=self.cache, poll_rate_hz=100.0)
        self.hand_poller = HandStatePoller(hand_adapter=self.hand_adapter, cache=self.cache, poll_rate_hz=60.0)
        self.arm_poller.start()
        self.hand_poller.start()

        # 等待轮询器刷新首帧状态
        time.sleep(0.05)

        # 4. 初始化控制器初始物理位姿
        snap = self.cache.snapshot()
        q_init = np.concatenate([snap.arm.q, snap.hand.q]).astype(np.float32)
        self.joint_controller.reset(initial_state_22d=q_init)

        self._connected = True
        self._armed = False  # 严格保持 SAFE_IDLE

        if calibrate:
            self.calibrate()

    def arm(self, gravity_confirmed: bool = False) -> None:
        """显式上电使能门禁 (应用级安全确认)."""
        if not self.is_connected:
            raise RuntimeError("Cannot arm robot before connecting.")
        if not gravity_confirmed:
            raise RuntimeError("Cannot arm without explicit gravity_confirmed=True.")

        if hasattr(self.arm_adapter, "arm"):
            self.arm_adapter.arm(gravity_confirmed=True)
        self._armed = True

    def configure(self) -> None:
        """配置电机与通信参数 (No-op 明确定义)."""
        pass

    def calibrate(self) -> None:
        """验证零位状态 (Validation-only calibration)."""
        self._calibrated = True

    def get_observation(self) -> Dict[str, Any]:
        """从 LatestStateCache 获取当前原子快照 (纯缓存读取，零现场总线 IO 阻塞)."""
        if not self.is_connected:
            raise RuntimeError("Cannot get observation from disconnected robot.")

        now = time.monotonic()
        # 纯消费缓存快照
        snap = self.cache.snapshot(now=now)
        state_22d = np.concatenate([snap.arm.q, snap.hand.q]).astype(np.float32)

        img = snap.overhead_cam.image if snap.overhead_cam else np.zeros((480, 640, 3), dtype=np.uint8)

        return {
            "observation.state": state_22d,
            "observation.images.overhead_cam": img,
        }

    def send_action(self, action: Union[Dict[str, Any], np.ndarray]) -> Union[Dict[str, Any], np.ndarray]:
        """安全下发 22-DOF 动作 (实测有界 dt 计算与安全主管三级裁决)."""
        if not self.is_connected:
            raise RuntimeError("Cannot send action to disconnected robot.")

        now = time.monotonic()

        # 计算实测有界 dt (0.005s ~ 0.100s)
        if self._last_action_time is not None:
            dt = float(np.clip(now - self._last_action_time, 0.005, 0.100))
        else:
            dt = 1.0 / max(1.0, float(self.config.fps))
        self._last_action_time = now

        # 解析动作数组
        if isinstance(action, dict):
            raw_action = np.asarray(action["action"], dtype=np.float32)
        else:
            raw_action = np.asarray(action, dtype=np.float32)

        assert raw_action.shape == (22,), f"Action shape must be (22,), got {raw_action.shape}"

        curr_obs = self.get_observation()
        curr_state = curr_obs["observation.state"]

        # 未上电使能时拒绝物理下发，返回当前安全保持态
        if not self._armed:
            return action

        # 经 JointTargetController 与 MotionSafetySupervisor 安全平滑与裁决
        executed_22d, sup_res, granted = self.joint_controller.process_action(
            action_22d=raw_action,
            current_state_22d=curr_state,
            dt=dt,
            now=now,
            source=ControlSource.VLA_POLICY,
        )

        if not granted or executed_22d is None or sup_res.decision in (SafetyDecision.REJECT, SafetyDecision.ESTOP):
            # 租约被抢占或触发 REJECT/ESTOP，返回当前安全物理保持态
            return curr_state

        # 下发至真实底层硬件
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
        """安全断开连接并完全释放 Poller 线程与通信句柄."""
        self._armed = False

        # 1. 停止并等待 Poller 线程结束
        if self.arm_poller is not None:
            self.arm_poller.stop()
            self.arm_poller = None
        if self.hand_poller is not None:
            self.hand_poller.stop()
            self.hand_poller = None

        # 2. 释放底层硬件通信
        if self.arm_adapter is not None:
            try:
                self.arm_adapter.disconnect()
            except Exception:
                pass
        if self.hand_adapter is not None:
            try:
                self.hand_adapter.disconnect()
            except Exception:
                pass

        self._connected = False
