"""TuinaSafetyLayer 模型自主推理安全拦截层 (Policy Safety Envelope Interceptor).

【核心安全防线】
1. 物理状态基准初始化: 必须以当前真实状态 current_state_22d 初始化平滑器，无基准时直接拒绝 (REJECT)；
2. 动作块平滑 (Action Chunk Smoothing): 避免神经网络离散输出导致机械臂关节阶跃震颤;
3. 软限位与速度硬截断 (Bounds & max_dq Clamping): 任何越界目标在到达硬件前强制夹紧;
4. 人类无条件抢占门禁 (Human Priority Preemption): 实时监听 ControlArbiter, 若检测到操作员介入立即阻断策略动作;
5. 姿态奇异与急停保护 (Singularity & E-stop Watchdog).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np

from Co_Teleop.safety.control_arbiter import ControlArbiter, ControlSource
from Co_Teleop.safety.motion_supervisor import MotionSafetySupervisor, SafetyDecision, SupervisorLimits, SupervisorResult


class TuinaSafetyLayer:
    """VLA / Diffusion Policy 模型自主推理安全拦截层."""

    def __init__(
        self,
        arbiter: Optional[ControlArbiter] = None,
        supervisor: Optional[MotionSafetySupervisor] = None,
        alpha_smooth: float = 0.6,
    ):
        self.arbiter = arbiter or ControlArbiter()
        self.supervisor = supervisor or MotionSafetySupervisor()
        self.alpha_smooth = float(np.clip(alpha_smooth, 0.05, 1.0))
        self._last_executed_22d: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None

    def reset(self, initial_state_22d: Optional[np.ndarray] = None) -> None:
        """重置安全层滤波状态."""
        if initial_state_22d is not None:
            self._last_executed_22d = np.asarray(initial_state_22d, dtype=np.float32).copy()
        else:
            self._last_executed_22d = None
        self._last_time = None

    def process_policy_chunk(
        self,
        policy_action_chunk: np.ndarray,
        current_state_22d: Optional[np.ndarray],
        dt: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], bool, str]:
        """处理模型预测的 22-DOF 动作块 (Action Chunk) 并返回下一拍安全执行目标.

        Args:
            policy_action_chunk: 模型输出的多步动作预测 (Chunk_Size, 22) 或单步 (22,)
            current_state_22d: 当前机器人 22 轴真实状态 (rad)
            dt: 控制周期 (若未提供则使用实测有界 dt)
            now: 当前单调时钟时间戳

        Returns:
            Tuple[Optional[np.ndarray], bool, str]:
                - safe_target_22d: 安全可执行的 22 维目标弧度 (被抢占或阻断时为 None)
                - granted: 是否成功获得控制权且通过安全裁决
                - reason: 状态原因描述
        """
        if now is None:
            now = time.monotonic()

        # 1. 动态测算实测有界 dt
        if dt is None:
            if self._last_time is not None:
                dt_measured = now - self._last_time
            else:
                dt_measured = 0.033
            dt = float(np.clip(dt_measured, 0.005, 0.100))
        else:
            dt = float(np.clip(dt, 0.005, 0.100))
        self._last_time = now

        # 2. 检查人类抢占租约
        active_source = self.arbiter.get_active_source(now=now)
        if active_source == ControlSource.HUMAN_TELEOP:
            return None, False, "Preempted by HUMAN_TELEOP"

        # 3. 申请策略控制租约
        granted = self.arbiter.request_lease(ControlSource.VLA_POLICY, now=now)
        if not granted:
            return None, False, "Control lease denied"

        # 4. 提取第 0 步动作 (或单步动作)
        if policy_action_chunk.ndim == 2:
            raw_target_22d = policy_action_chunk[0]
        else:
            raw_target_22d = policy_action_chunk

        assert raw_target_22d.shape == (22,), f"Target action shape must be (22,), got {raw_target_22d.shape}"

        # 5. 首帧物理基准安全对齐
        if self._last_executed_22d is None:
            if current_state_22d is not None:
                self._last_executed_22d = np.asarray(current_state_22d, dtype=np.float32).copy()
            else:
                return None, False, "REJECT: No current_state_22d available to initialize safety layer"

        curr_state = np.asarray(current_state_22d, dtype=np.float32) if current_state_22d is not None else self._last_executed_22d

        # 6. 平滑滤波
        smoothed_target = (
            self.alpha_smooth * raw_target_22d + (1.0 - self.alpha_smooth) * self._last_executed_22d
        ).astype(np.float32)

        # 7. 经过安全主管裁决与软限位裁剪
        sup_res = self.supervisor.supervise_22d(
            action_22d=smoothed_target,
            current_state_22d=curr_state,
            dt=dt,
            source=ControlSource.VLA_POLICY,
        )

        clamped_22d = np.concatenate([sup_res.clamped_arm_q, sup_res.clamped_hand_q]).astype(np.float32)

        if sup_res.decision in (SafetyDecision.REJECT, SafetyDecision.ESTOP):
            self._last_executed_22d = curr_state.copy()
            return None, False, f"Supervisor {sup_res.decision.value}: {sup_res.reason}"

        # 成功执行
        self._last_executed_22d = clamped_22d.copy()
        return clamped_22d, True, f"Executed ({sup_res.decision.value}: {sup_res.reason})"
