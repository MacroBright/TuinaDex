"""异步状态轮询器与原子状态缓存 (State Poller & LatestStateCache).

核心职责：
1. 运行独立的 ArmStatePoller 与 HandStatePoller 线程，非阻塞获取底层总线真实数据；
2. 维护线程安全的 LatestStateCache；
3. 提供原子快照 snapshot(now=t)，并根据各传感器最大允许时延 (max_source_age_ms) 进行新鲜度与健康判定。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Optional, Tuple
import numpy as np

from Co_Teleop.recording.teleop_frame import (
    AcupointObservation,
    ArmObservation,
    CameraFrame,
    HandObservation,
)


@dataclass
class FreshnessLimits:
    """各数据源最大允许时延门限 (毫秒)."""

    max_arm_age_ms: float = 80.0          # 机械臂 CAN 状态最大时延 (20~50Hz)
    max_hand_age_ms: float = 100.0        # 灵巧手串口状态最大时延 (30Hz)
    max_cam_age_ms: float = 120.0         # 工业相机图像最大时延 (30Hz)
    max_acupoint_age_ms: float = 300.0    # 穴位神经网络检测最大时延 (10~20Hz)


@dataclass
class StateSnapshot:
    """原子状态快照."""

    now: float
    arm: ArmObservation
    hand: HandObservation
    overhead_cam: Optional[CameraFrame]
    acupoints: Optional[AcupointObservation]

    # 时延与新鲜度统计
    arm_age_ms: float
    hand_age_ms: float
    cam_age_ms: Optional[float]
    acupoint_age_ms: Optional[float]

    is_fresh: bool                        # 所有必要源均在时限内
    is_degraded: bool                     # 部分源出现延迟超限
    degraded_reasons: list[str]


class LatestStateCache:
    """线程安全的多源状态缓存."""

    def __init__(self, limits: Optional[FreshnessLimits] = None):
        self._lock = threading.RLock()
        self.limits = limits or FreshnessLimits()

        # 默认初始空观测
        self._arm_obs = ArmObservation(
            q=np.zeros(6, dtype=np.float32),
            dq=np.zeros(6, dtype=np.float32),
            current=np.zeros(6, dtype=np.float32),
            timestamp=0.0,
        )
        self._hand_obs = HandObservation(
            q=np.zeros(16, dtype=np.float32),
            currents=np.zeros(16, dtype=np.float32),
            timestamp=0.0,
        )
        self._overhead_cam: Optional[CameraFrame] = None
        self._acupoints: Optional[AcupointObservation] = None

    def update_arm(self, obs: ArmObservation) -> None:
        with self._lock:
            self._arm_obs = obs

    def update_hand(self, obs: HandObservation) -> None:
        with self._lock:
            self._hand_obs = obs

    def update_camera(self, cam_frame: CameraFrame) -> None:
        with self._lock:
            self._overhead_cam = cam_frame

    def update_acupoints(self, acupoints: AcupointObservation) -> None:
        with self._lock:
            self._acupoints = acupoints

    def snapshot(self, now: Optional[float] = None) -> StateSnapshot:
        """获取当前时刻一致性快照并计算新鲜度."""
        if now is None:
            now = time.monotonic()

        with self._lock:
            arm = self._arm_obs
            hand = self._hand_obs
            cam = self._overhead_cam
            acu = self._acupoints

        arm_age = (now - arm.timestamp) * 1000.0 if arm.timestamp > 0 else 999999.0
        hand_age = (now - hand.timestamp) * 1000.0 if hand.timestamp > 0 else 999999.0
        cam_age = (now - cam.timestamp) * 1000.0 if cam and cam.timestamp > 0 else None
        acu_age = (now - acu.timestamp) * 1000.0 if acu and acu.timestamp > 0 else None

        reasons: list[str] = []
        if arm_age > self.limits.max_arm_age_ms:
            reasons.append(f"Arm state stale ({arm_age:.1f}ms > {self.limits.max_arm_age_ms}ms)")
        if hand_age > self.limits.max_hand_age_ms:
            reasons.append(f"Hand state stale ({hand_age:.1f}ms > {self.limits.max_hand_age_ms}ms)")
        if cam_age is not None and cam_age > self.limits.max_cam_age_ms:
            reasons.append(f"Camera frame stale ({cam_age:.1f}ms > {self.limits.max_cam_age_ms}ms)")
        if acu_age is not None and acu_age > self.limits.max_acupoint_age_ms:
            reasons.append(f"Acupoint det stale ({acu_age:.1f}ms > {self.limits.max_acupoint_age_ms}ms)")

        is_fresh = (len(reasons) == 0) and (arm.timestamp > 0) and (hand.timestamp > 0)
        is_degraded = not is_fresh

        return StateSnapshot(
            now=now,
            arm=arm,
            hand=hand,
            overhead_cam=cam,
            acupoints=acu,
            arm_age_ms=arm_age,
            hand_age_ms=hand_age,
            cam_age_ms=cam_age,
            acupoint_age_ms=acu_age,
            is_fresh=is_fresh,
            is_degraded=is_degraded,
            degraded_reasons=reasons,
        )


class ArmStatePoller:
    """机械臂状态异步轮询线程."""

    def __init__(self, arm_adapter: Any, cache: LatestStateCache, poll_rate_hz: float = 30.0):
        self.arm_adapter = arm_adapter
        self.cache = cache
        self.interval = 1.0 / max(1.0, poll_rate_hz)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="ArmStatePoller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        while self._running:
            t_start = time.monotonic()
            try:
                # 从 arm_adapter 提取当前真实弧度与状态
                q, dq, current, flags = self._query_adapter()
                obs = ArmObservation(
                    q=q,
                    dq=dq,
                    current=current,
                    flags=flags,
                    timestamp=t_start,
                )
                self.cache.update_arm(obs)
            except Exception as e:
                # 出现异常记录但不崩溃轮询线程
                pass

            elapsed = time.monotonic() - t_start
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _query_adapter(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        # 兼容 RealArmAdapter, SimArmAdapter, NoDriveArmAdapter
        q = np.zeros(6, dtype=np.float32)
        dq = np.zeros(6, dtype=np.float32)
        current = np.zeros(6, dtype=np.float32)
        flags = 0

        if hasattr(self.arm_adapter, "get_state"):
            state = self.arm_adapter.get_state()
            if isinstance(state, dict):
                q = np.asarray(state.get("q", np.zeros(6)), dtype=np.float32)
                dq = np.asarray(state.get("dq", np.zeros(6)), dtype=np.float32)
                current = np.asarray(state.get("current", np.zeros(6)), dtype=np.float32)
                flags = int(state.get("flags", 0))
            elif hasattr(state, "q"):
                q = np.asarray(state.q, dtype=np.float32)
                dq = np.asarray(getattr(state, "dq", np.zeros(6)), dtype=np.float32)
                current = np.asarray(getattr(state, "current", np.zeros(6)), dtype=np.float32)
                flags = int(getattr(state, "flags", 0))
        elif hasattr(self.arm_adapter, "current_q_rad"):
            q = np.asarray(self.arm_adapter.current_q_rad, dtype=np.float32)

        return q, dq, current, flags


class HandStatePoller:
    """灵巧手状态异步轮询线程."""

    def __init__(self, hand_adapter: Any, cache: LatestStateCache, poll_rate_hz: float = 30.0):
        self.hand_adapter = hand_adapter
        self.cache = cache
        self.interval = 1.0 / max(1.0, poll_rate_hz)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="HandStatePoller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        while self._running:
            t_start = time.monotonic()
            try:
                q, currents = self._query_adapter()
                obs = HandObservation(
                    q=q,
                    currents=currents,
                    timestamp=t_start,
                )
                self.cache.update_hand(obs)
            except Exception:
                pass

            elapsed = time.monotonic() - t_start
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _query_adapter(self) -> Tuple[np.ndarray, np.ndarray]:
        q = np.zeros(16, dtype=np.float32)
        currents = np.zeros(16, dtype=np.float32)

        if hasattr(self.hand_adapter, "get_angles"):
            angles = self.hand_adapter.get_angles()
            if angles is not None:
                q = np.asarray(angles, dtype=np.float32)
        elif hasattr(self.hand_adapter, "current_angles"):
            q = np.asarray(self.hand_adapter.current_angles, dtype=np.float32)

        if hasattr(self.hand_adapter, "get_currents"):
            curr = self.hand_adapter.get_currents()
            if curr is not None:
                currents = np.asarray(curr, dtype=np.float32)

        return q, currents
