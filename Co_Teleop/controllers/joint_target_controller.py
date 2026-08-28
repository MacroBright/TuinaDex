"""JointTargetController 关节目标动作平滑控制器 (含首帧基准防护与安全裁决).

职责：
1. 接收来自 VLA / Diffusion Policy / Replay 的 22 维关节目标弧度 (Action 22D)；
2. 检查并向 ControlArbiter 申请控制租约 (优先保证人类遥操最高抢占权)；
3. 以机器人当前真实物理状态 (current_state_22d) 初始化平滑滤波器，杜绝第一帧动作跳变；
4. 若无有效当前物理状态，直接拒绝 (REJECT) 并返回安全保持，严禁盲目执行未对齐 Action；
5. 在有界实测 dt (0.005s ~ 0.100s) 下执行指数移动平均 (EMA) 平滑与 MotionSafetySupervisor 安全截断。
"""

from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np

from Co_Teleop.safety.control_arbiter import ControlArbiter, ControlSource
from Co_Teleop.safety.motion_supervisor import MotionSafetySupervisor, SafetyDecision, SupervisorResult


class JointTargetController:
    """22-DOF 关节目标控制器 (VLA Policy & Replay 执行专用)."""

    def __init__(
        self,
        arbiter: ControlArbiter,
        supervisor: Optional[MotionSafetySupervisor] = None,
        alpha_arm: float = 0.5,
        alpha_hand: float = 0.8,
    ):
        self.arbiter = arbiter
        self.supervisor = supervisor or MotionSafetySupervisor()
        self.alpha_arm = float(np.clip(alpha_arm, 0.05, 1.0))
        self.alpha_hand = float(np.clip(alpha_hand, 0.05, 1.0))

        self._filtered_arm_q: Optional[np.ndarray] = None
        self._filtered_hand_q: Optional[np.ndarray] = None
        self._last_process_time: Optional[float] = None

    def reset(self, initial_state_22d: Optional[np.ndarray] = None) -> None:
        """重置平滑滤波器状态 (建议传入机器人当前物理弧度)."""
        if initial_state_22d is not None:
            init_22d = np.asarray(initial_state_22d, dtype=np.float32)
            assert init_22d.shape == (22,), f"Initial state shape must be (22,), got {init_22d.shape}"
            self._filtered_arm_q = init_22d[:6].copy()
            self._filtered_hand_q = init_22d[6:].copy()
        else:
            self._filtered_arm_q = None
            self._filtered_hand_q = None
        self._last_process_time = None

    def process_action(
        self,
        action_22d: np.ndarray,
        current_state_22d: Optional[np.ndarray],
        dt: Optional[float] = None,
        now: Optional[float] = None,
        source: ControlSource = ControlSource.VLA_POLICY,
    ) -> Tuple[Optional[np.ndarray], SupervisorResult, bool]:
        """对 22 维策略动作进行租约仲裁、首帧状态对齐平滑与安全主管限幅.

        Args:
            action_22d: 策略输出目标 (22,) 包含 6 臂 + 16 手
            current_state_22d: 当前机器人真实物理角度 (22,)
            dt: 控制周期 (若未提供则通过实测单调时钟计算)
            now: 单调时钟时间戳
            source: 控制来源 (VLA_POLICY 或 DATASET_REPLAY)

        Returns:
            Tuple[Optional[np.ndarray], SupervisorResult, bool]:
                - executed_22d: 最终安全执行的 22 维弧度 (未获租约或被 REJECT 时返回保持态)
                - supervisor_result: 安全主管审查结果
                - lease_granted: 是否成功获得控制权
        """
        if now is None:
            now = time.monotonic()

        # 1. 动态测算实测有界 dt (防止线程阻塞后速度限幅失真)
        if dt is None:
            if self._last_process_time is not None:
                dt_measured = now - self._last_process_time
            else:
                dt_measured = 0.033
            dt = float(np.clip(dt_measured, 0.005, 0.100))
        else:
            dt = float(np.clip(dt, 0.005, 0.100))
        self._last_process_time = now

        # 2. 向 ControlArbiter 申请控制租约 (检查是否被 HUMAN_TELEOP 抢占)
        granted = self.arbiter.request_lease(source, now=now)
        if not granted:
            hold_arm = current_state_22d[:6].copy() if current_state_22d is not None else np.zeros(6, dtype=np.float32)
            hold_hand = current_state_22d[6:].copy() if current_state_22d is not None else np.zeros(16, dtype=np.float32)
            empty_sup = SupervisorResult(
                decision=SafetyDecision.REJECT,
                is_safe=False,
                clamped_arm_q=hold_arm,
                clamped_hand_q=hold_hand,
                arm_clamped=True,
                hand_clamped=True,
                source=source,
                reason=f"Lease denied: active source is {self.arbiter.get_active_source(now=now).value}",
            )
            return None, empty_sup, False

        raw_act = np.asarray(action_22d, dtype=np.float32)
        assert raw_act.shape == (22,), f"Action shape must be (22,), got {raw_act.shape}"
        arm_raw = raw_act[:6]
        hand_raw = raw_act[6:]

        # 3. 初始帧安全对齐防跳变
        if self._filtered_arm_q is None or self._filtered_hand_q is None:
            if current_state_22d is not None:
                # 严格以真实物理角度初始化平滑器
                self._filtered_arm_q = current_state_22d[:6].copy()
                self._filtered_hand_q = current_state_22d[6:].copy()
            else:
                # 无当前有效状态基准，直接拒绝动作
                empty_sup = SupervisorResult(
                    decision=SafetyDecision.REJECT,
                    is_safe=False,
                    clamped_arm_q=np.zeros(6, dtype=np.float32),
                    clamped_hand_q=np.zeros(16, dtype=np.float32),
                    arm_clamped=True,
                    hand_clamped=True,
                    source=source,
                    reason="REJECT: No current_state_22d available to initialize smoothing filter",
                )
                return None, empty_sup, False

        # 4. 指数移动平均 (EMA) 平滑滤波
        self._filtered_arm_q = (self.alpha_arm * arm_raw + (1.0 - self.alpha_arm) * self._filtered_arm_q).astype(np.float32)
        self._filtered_hand_q = (self.alpha_hand * hand_raw + (1.0 - self.alpha_hand) * self._filtered_hand_q).astype(np.float32)

        smoothed_22d = np.concatenate([self._filtered_arm_q, self._filtered_hand_q]).astype(np.float32)

        # 5. 经过 MotionSafetySupervisor 进行三级安全裁决
        sup_result = self.supervisor.supervise_22d(
            action_22d=smoothed_22d,
            current_state_22d=current_state_22d,
            dt=dt,
            source=source,
        )

        # 6. 若发生严重 REJECT 或 ESTOP，重置平滑器为安全当前态
        if sup_result.decision in (SafetyDecision.REJECT, SafetyDecision.ESTOP):
            if current_state_22d is not None:
                self._filtered_arm_q = current_state_22d[:6].copy()
                self._filtered_hand_q = current_state_22d[6:].copy()
            executed_22d = np.concatenate([sup_result.clamped_arm_q, sup_result.clamped_hand_q]).astype(np.float32)
            return executed_22d, sup_result, False

        # 正常或夹紧执行
        self._filtered_arm_q = sup_result.clamped_arm_q.copy()
        self._filtered_hand_q = sup_result.clamped_hand_q.copy()
        executed_22d = np.concatenate([sup_result.clamped_arm_q, sup_result.clamped_hand_q]).astype(np.float32)

        return executed_22d, sup_result, True
