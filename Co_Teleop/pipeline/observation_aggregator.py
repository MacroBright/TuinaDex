"""ObservationAggregator 30Hz 观测聚合器.

职责：
1. 运行 30Hz 主时钟 Tick，从 LatestStateCache 获取各数据源原子快照；
2. 组合 6 轴机械臂真实角度与 16 轴灵巧手弧度为 22-DOF observation_state_22d；
3. 结合当前动作目标 (action_22d)、人类笛卡尔意图 (cartesian_cmd)、工业相机图像与穴位检测；
4. 根据离合/暂停/视觉时延判定并生成统一强类型 TeleopFrame。
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from Co_Teleop.adapters.state_poller import LatestStateCache, StateSnapshot
from Co_Teleop.recording.teleop_frame import (
    AcupointObservation,
    CameraFrame,
    FrameValidity,
    TaskMetadata,
    TeleopFrame,
)


class ObservationAggregator:
    """观测与时空快照聚合器."""

    def __init__(self, cache: LatestStateCache):
        self.cache = cache
        self._frame_id = 0
        self._is_paused = False
        self._is_hand_locked = False

    def set_pause(self, paused: bool) -> None:
        """设置空格离合暂停状态."""
        self._is_paused = paused

    def set_hand_locked(self, locked: bool) -> None:
        """设置灵巧手 L 键姿态锁定状态."""
        self._is_hand_locked = locked

    def aggregate(
        self,
        action_22d: np.ndarray,
        cartesian_cmd: np.ndarray,
        task: Optional[TaskMetadata] = None,
        now: Optional[float] = None,
        extra_validity: Optional[FrameValidity] = None,
    ) -> TeleopFrame:
        """在当前时刻聚合生成一帧标准 TeleopFrame.

        Args:
            action_22d: 当前下发的 22 维执行目标弧度
            cartesian_cmd: 当前人类操作员的 6 维笛卡尔线速度与角速度
            task: 当前任务元数据
            now: 主时钟单调时间戳
            extra_validity: 外部覆盖的有效性 (如 ESTOP)

        Returns:
            不可变且完成时空对齐的 TeleopFrame
        """
        snapshot: StateSnapshot = self.cache.snapshot(now=now)
        self._frame_id += 1

        # 1. 组合 22-DOF 观测状态 (前 6 机械臂, 后 16 灵巧手)
        arm_q = snapshot.arm.q
        hand_q = snapshot.hand.q
        obs_22d = np.concatenate([arm_q, hand_q]).astype(np.float32)

        # 2. 判定有效性
        if extra_validity is not None:
            validity = extra_validity
        elif self._is_paused:
            validity = FrameValidity.PAUSED
        elif self._is_hand_locked:
            validity = FrameValidity.HAND_LOCKED
        elif snapshot.is_degraded:
            validity = FrameValidity.VISION_DEGRADED
        else:
            validity = FrameValidity.VALID

        frame = TeleopFrame(
            frame_id=self._frame_id,
            timestamp=snapshot.now,
            arm=snapshot.arm,
            hand=snapshot.hand,
            overhead_cam=snapshot.overhead_cam,
            observation_state_22d=obs_22d,
            action_22d=action_22d,
            cartesian_cmd=cartesian_cmd,
            acupoints=snapshot.acupoints,
            validity=validity,
            task=task or TaskMetadata(),
        )

        return frame
