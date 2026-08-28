"""JointTargetController 关节目标控制器 (专用于 VLA / Policy 22-DOF 动作输入).

职责：
1. 接收 AI 策略网络或外部规划器输出的 22-DOF 动作 (前 6 臂 + 后 16 手)；
2. 执行平滑滤波 (指数滑动平均 EMA 或限速插值)，防止神经网络分块输出导致的机械抖动；
3. 向 ControlArbiter 请求 VLA_POLICY 租约，获得授权后提交给 MotionSafetySupervisor 进行终极合规校验；
4. 校验通过后分发给真实硬件适配器 (RealArmAdapter / LeapHandAdapter)。
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from Co_Teleop.safety.control_arbiter import ControlArbiter, ControlSource
from Co_Teleop.safety.motion_supervisor import MotionSafetySupervisor, SupervisorResult


class JointTargetController:
    """关节目标控制器."""

    def __init__(
        self,
        arbiter: ControlArbiter,
        supervisor: MotionSafetySupervisor,
        alpha_arm: float = 0.5,           # 机械臂目标低通平滑系数 (0~1, 1 为直接直通)
        alpha_hand: float = 0.8,          # 灵巧手目标低通平滑系数
    ):
        self.arbiter = arbiter
        self.supervisor = supervisor
        self.alpha_arm = float(np.clip(alpha_arm, 0.05, 1.0))
        self.alpha_hand = float(np.clip(alpha_hand, 0.05, 1.0))

        self._filtered_arm_q: Optional[np.ndarray] = None
        self._filtered_hand_q: Optional[np.ndarray] = None

    def reset(self, initial_state_22d: Optional[np.ndarray] = None) -> None:
        """重置滤波器内部状态."""
        if initial_state_22d is not None:
            state = np.asarray(initial_state_22d, dtype=np.float32)
            self._filtered_arm_q = state[:6].copy()
            self._filtered_hand_q = state[6:].copy()
        else:
            self._filtered_arm_q = None
            self._filtered_hand_q = None

    def process_action(
        self,
        action_22d: np.ndarray,
        current_state_22d: Optional[np.ndarray] = None,
        dt: float = 0.050,
        now: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], SupervisorResult, bool]:
        """处理一帧 22-DOF 策略动作.

        Args:
            action_22d: 策略输出的 22 维目标关节弧度 (6 臂 + 16 手)
            current_state_22d: 当前机器人 22 维真实状态
            dt: 控制周期时间间隔 (秒)
            now: 当前单调时间戳

        Returns:
            (executed_22d, supervisor_result, lease_granted)
            若未获得租约，executed_22d 返回 None。
        """
        action = np.asarray(action_22d, dtype=np.float32)
        assert action.shape == (22,), f"Action shape must be (22,), got {action.shape}"

        # 1. 尝试向仲裁器申请/刷新 VLA_POLICY 租约
        lease_granted = self.arbiter.request_lease(ControlSource.VLA_POLICY, now=now)
        if not lease_granted:
            # 租约被抢占 (例如人类正在操作或急停中)
            res = SupervisorResult(
                is_safe=False,
                clamped_arm_q=action[:6],
                clamped_hand_q=action[6:],
                arm_clamped=False,
                hand_clamped=False,
                reason="Control lease denied: preempted by human teleop or emergency stop",
            )
            return None, res, False

        arm_raw = action[:6]
        hand_raw = action[6:]

        # 2. 低通平滑滤波 (防止 action chunk 边缘阶跃)
        if self._filtered_arm_q is None:
            self._filtered_arm_q = arm_raw.copy()
        else:
            self._filtered_arm_q = self.alpha_arm * arm_raw + (1.0 - self.alpha_arm) * self._filtered_arm_q

        if self._filtered_hand_q is None:
            self._filtered_hand_q = hand_raw.copy()
        else:
            self._filtered_hand_q = self.alpha_hand * hand_raw + (1.0 - self.alpha_hand) * self._filtered_hand_q

        smoothed_22d = np.concatenate([self._filtered_arm_q, self._filtered_hand_q])

        # 3. 提交给 MotionSafetySupervisor 进行全套硬/软限位与跳变截断
        sup_result = self.supervisor.supervise_22d(
            action_22d=smoothed_22d,
            current_state_22d=current_state_22d,
            dt=dt,
        )

        executed_22d = np.concatenate([sup_result.clamped_arm_q, sup_result.clamped_hand_q])
        # 滤波状态回填为真正合规截断后的值
        self._filtered_arm_q = sup_result.clamped_arm_q.copy()
        self._filtered_hand_q = sup_result.clamped_hand_q.copy()

        return executed_22d, sup_result, True
