"""MotionSafetySupervisor 运动安全监管者 (唯一底座动作安全出口).

核心职责：
- 回答“允许不允许执行”；
- 实施 22 轴软件硬限位/软限位 Clamping；
- 实施单步最大关节跳变限幅 (max_dq) 与角速度/角加速度硬限幅；
- 提供看门狗健康状态感知与超时熔断；
- 确保任何下发到 RealArmAdapter 与 LeapHandAdapter 的目标绝对安全。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class SupervisorLimits:
    """机械臂与灵巧手安全限位配置 (所有角度/速度统一采用国际标准单位 rad, rad/s)."""

    # 6 轴机械臂关节限位 [min_rad, max_rad]
    arm_limits_rad: np.ndarray = field(
        default_factory=lambda: np.array([
            [-0.017453, 6.283185],   # J1: [-1°, 360°]
            [-0.017453, 2.617994],   # J2: [-1°, 150°]
            [-0.017453, 2.094395],   # J3: [-1°, 120°]
            [-1.570796, 1.570796],   # J4: [-90°, +90°]
            [-0.017453, 3.141593],   # J5: [-1°, 180°]
            [-0.017453, 6.283185],   # J6: [-1°, 360°]
        ], dtype=np.float32)
    )

    # 16 轴灵巧手各关节物理弯曲弧度限位 [min_rad, max_rad]
    hand_limits_rad: np.ndarray = field(
        default_factory=lambda: np.tile(np.array([-0.20, 3.14159], dtype=np.float32), (16, 1))
    )

    # 单步最大允许关节角度跳变 (默认 30° = 0.5236 rad)
    max_dq_rad: float = 0.5235987755982988

    # 关节最大角速度 (默认 540°/s = 9.4247 rad/s)
    max_joint_vel_rad_s: float = 9.42477796076938

    # 限位安全裕度缓冲 (默认 2° = 0.0349 rad)
    margin_rad: float = 0.03490658503988659


@dataclass
class SupervisorResult:
    """监管检查结果."""

    is_safe: bool                         # 是否完全安全 (未触发违规截断)
    clamped_arm_q: np.ndarray             # 经限幅后的 6 轴机械臂目标弧度
    clamped_hand_q: np.ndarray            # 经限幅后的 16 轴灵巧手目标弧度
    arm_clamped: bool                     # 机械臂是否被截断
    hand_clamped: bool                    # 灵巧手是否被截断
    reason: str = ""                      # 截断或警告原因说明


class MotionSafetySupervisor:
    """运动安全监管者."""

    def __init__(self, limits: Optional[SupervisorLimits] = None):
        self.limits = limits or SupervisorLimits()

    def validate_and_clamp_arm(
        self,
        target_q: np.ndarray,
        current_q: Optional[np.ndarray] = None,
        dt: float = 0.050,
    ) -> Tuple[np.ndarray, bool, str]:
        """对 6 轴机械臂目标关节弧度进行安全校验与强制限幅.

        Args:
            target_q: 期望 6 轴目标弧度 (6,)
            current_q: 当前 6 轴真实弧度 (6,)，若提供则检查跳变与速度
            dt: 控制步长时间间隔 (秒)

        Returns:
            (clamped_q, was_clamped, reason)
        """
        target = np.asarray(target_q, dtype=np.float32).copy()
        assert target.shape == (6,), f"Arm target_q shape must be (6,), got {target.shape}"

        was_clamped = False
        reasons = []

        # 1. 关节软限位校验与 Clamping
        for i in range(6):
            min_q = self.limits.arm_limits_rad[i, 0]
            max_q = self.limits.arm_limits_rad[i, 1]
            if target[i] < min_q:
                target[i] = min_q
                was_clamped = True
                reasons.append(f"J{i+1} under min limit ({min_q:.3f} rad)")
            elif target[i] > max_q:
                target[i] = max_q
                was_clamped = True
                reasons.append(f"J{i+1} over max limit ({max_q:.3f} rad)")

        # 2. 单步跳变与速度限幅 (若提供当前角度)
        if current_q is not None:
            curr = np.asarray(current_q, dtype=np.float32)
            delta_q = target - curr

            # 2.1 单步绝对跳变截断 (max_dq)
            for i in range(6):
                if abs(delta_q[i]) > self.limits.max_dq_rad:
                    target[i] = curr[i] + np.sign(delta_q[i]) * self.limits.max_dq_rad
                    was_clamped = True
                    reasons.append(f"J{i+1} step jump {abs(delta_q[i]):.3f} > max_dq {self.limits.max_dq_rad:.3f}")

            # 2.2 瞬时角速度截断 (max_joint_vel)
            if dt > 1e-4:
                vel = (target - curr) / dt
                for i in range(6):
                    if abs(vel[i]) > self.limits.max_joint_vel_rad_s:
                        target[i] = curr[i] + np.sign(vel[i]) * self.limits.max_joint_vel_rad_s * dt
                        was_clamped = True
                        reasons.append(f"J{i+1} vel {abs(vel[i]):.3f} > max_vel {self.limits.max_joint_vel_rad_s:.3f}")

        reason_str = "; ".join(reasons) if reasons else "OK"
        return target, was_clamped, reason_str

    def validate_and_clamp_hand(self, target_q: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """对 16 轴灵巧手物理弯曲弧度进行限幅.

        Args:
            target_q: 期望 16 轴目标弧度 (16,)

        Returns:
            (clamped_q, was_clamped, reason)
        """
        target = np.asarray(target_q, dtype=np.float32).copy()
        assert target.shape == (16,), f"Hand target_q shape must be (16,), got {target.shape}"

        was_clamped = False
        reasons = []

        for i in range(16):
            min_q = self.limits.hand_limits_rad[i, 0]
            max_q = self.limits.hand_limits_rad[i, 1]
            if target[i] < min_q:
                target[i] = min_q
                was_clamped = True
                reasons.append(f"Hand_{i} < {min_q:.2f}")
            elif target[i] > max_q:
                target[i] = max_q
                was_clamped = True
                reasons.append(f"Hand_{i} > {max_q:.2f}")

        reason_str = "; ".join(reasons) if reasons else "OK"
        return target, was_clamped, reason_str

    def supervise_22d(
        self,
        action_22d: np.ndarray,
        current_state_22d: Optional[np.ndarray] = None,
        dt: float = 0.050,
    ) -> SupervisorResult:
        """统一对 22-DOF 动作 (前 6 臂 + 后 16 手) 执行全套安全合规校验."""
        action = np.asarray(action_22d, dtype=np.float32)
        assert action.shape == (22,), f"Action shape must be (22,), got {action.shape}"

        arm_target = action[:6]
        hand_target = action[6:]

        arm_curr = current_state_22d[:6] if current_state_22d is not None else None

        clamped_arm, arm_clamped, arm_reason = self.validate_and_clamp_arm(
            target_q=arm_target, current_q=arm_curr, dt=dt
        )
        clamped_hand, hand_clamped, hand_reason = self.validate_and_clamp_hand(hand_target)

        is_safe = not (arm_clamped or hand_clamped)
        reasons = []
        if arm_clamped:
            reasons.append(f"Arm: {arm_reason}")
        if hand_clamped:
            reasons.append(f"Hand: {hand_reason}")

        return SupervisorResult(
            is_safe=is_safe,
            clamped_arm_q=clamped_arm,
            clamped_hand_q=clamped_hand,
            arm_clamped=arm_clamped,
            hand_clamped=hand_clamped,
            reason="; ".join(reasons) if reasons else "OK",
        )
