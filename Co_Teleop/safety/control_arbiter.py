"""ControlArbiter 控制仲裁器与 MotionLease 运动租约机制.

核心规则：
1. 任何时刻只能有一个动作源持有物理机械臂的运动租约；
2. HUMAN_TELEOP (人类手势遥操) 享有最高抢占权，一旦检测到操作员意图即刻无条件收回 Policy 租约；
3. 租约具备默认 50ms 心跳超时机制，当推理进程掉帧或失联时自动过期并切入阻尼安全保持。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Optional


class ControlSource(str, Enum):
    """动作控制源枚举."""

    IDLE = "IDLE"                         # 无活动控制源 (安全待机态)
    HUMAN_TELEOP = "HUMAN_TELEOP"         # 人类操作员视觉手势遥操 (最高抢占权)
    VLA_POLICY = "VLA_POLICY"             # 策略网络自主推理 (SmolVLA / ACT / Diffusion)
    SCRIPT = "SCRIPT"                     # 预设脚本 / 复位回归动作 (如 Ready / Home 姿态)


@dataclass
class MotionLease:
    """运动租约数据结构."""

    holder: ControlSource
    granted_at: float                     # 租约授权单调时间 (time.monotonic())
    duration_s: float = 0.050             # 租约默认有效时长 (默认 50ms = 20Hz 容错心跳)

    def is_valid(self, now: Optional[float] = None) -> bool:
        """检查租约是否在有效期内."""
        if self.holder == ControlSource.IDLE:
            return False
        if now is None:
            now = time.monotonic()
        return (now - self.granted_at) <= self.duration_s

    @property
    def age_ms(self) -> float:
        """获取当前租约已持续时间 (毫秒)."""
        return (time.monotonic() - self.granted_at) * 1000.0


class ControlArbiter:
    """控制仲裁器 (线程安全)."""

    def __init__(self, default_lease_duration_s: float = 0.050):
        self._lock = threading.Lock()
        self._default_duration_s = default_lease_duration_s
        self._current_lease = MotionLease(
            holder=ControlSource.IDLE,
            granted_at=0.0,
            duration_s=default_lease_duration_s,
        )

    def request_lease(
        self,
        source: ControlSource,
        duration_s: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """请求获取或刷新运动控制租约.

        - HUMAN_TELEOP: 无条件直接抢占其他任何源；
        - SCRIPT: 仅在当前空闲、已过期或同为 SCRIPT 时授予；
        - VLA_POLICY: 仅在当前空闲、已过期或同为 VLA_POLICY 时授予 (人类遥操一动即被抢占).
        """
        if now is None:
            now = time.monotonic()
        if duration_s is None:
            duration_s = self._default_duration_s

        with self._lock:
            current_holder = self._current_lease.holder
            is_valid = self._current_lease.is_valid(now)

            # 1. 人类遥操享有绝对抢占权
            if source == ControlSource.HUMAN_TELEOP:
                self._current_lease = MotionLease(
                    holder=ControlSource.HUMAN_TELEOP,
                    granted_at=now,
                    duration_s=duration_s,
                )
                return True

            # 2. 如果当前租约已过期，任何有效源均可获取
            if not is_valid or current_holder == ControlSource.IDLE:
                self._current_lease = MotionLease(
                    holder=source,
                    granted_at=now,
                    duration_s=duration_s,
                )
                return True

            # 3. 如果当前租约仍有效且属于同一请求源，直接刷新心跳
            if current_holder == source:
                self._current_lease = MotionLease(
                    holder=source,
                    granted_at=now,
                    duration_s=duration_s,
                )
                return True

            # 4. 其他情况 (例如 VLA_POLICY 试图抢占正在活动的 HUMAN_TELEOP) 拒绝
            return False

    def release_lease(self, source: ControlSource) -> None:
        """主动释放租约."""
        with self._lock:
            if self._current_lease.holder == source:
                self._current_lease = MotionLease(
                    holder=ControlSource.IDLE,
                    granted_at=time.monotonic(),
                    duration_s=self._default_duration_s,
                )

    def force_idle(self) -> None:
        """强制重置仲裁器为空闲态 (用于急停或看门狗熔断)."""
        with self._lock:
            self._current_lease = MotionLease(
                holder=ControlSource.IDLE,
                granted_at=time.monotonic(),
                duration_s=self._default_duration_s,
            )

    def validate_action_source(self, source: ControlSource, now: Optional[float] = None) -> bool:
        """校验给定的动作源当前是否拥有合法执行权."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            if not self._current_lease.is_valid(now):
                return False
            return self._current_lease.holder == source

    def get_active_source(self, now: Optional[float] = None) -> ControlSource:
        """获取当前持有有效租约的控制源，若已过期则返回 IDLE."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            if self._current_lease.is_valid(now):
                return self._current_lease.holder
            return ControlSource.IDLE

    def get_lease_info(self) -> MotionLease:
        """获取当前租约快照."""
        with self._lock:
            return MotionLease(
                holder=self._current_lease.holder,
                granted_at=self._current_lease.granted_at,
                duration_s=self._current_lease.duration_s,
            )
