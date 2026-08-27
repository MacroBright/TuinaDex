"""scripts/teleop/watchdog.py — 视觉遥操分级看门狗 (spec §6.2, TASK-16).

职责边界 (P0-②): 陈旧命令判定归控制层 (CartesianController.step 的 cmd_ts
单调期限检查), 本看门狗**不接收 cmd_ts** — 只对视觉信号分级:
手丢失 / 低置信 / 深度无效 / 腕跳变, 产出 (action, velocity_scale).
禁止无限保持上一帧命令 (持续丢失 → 停; 严重 → e_stop 请求).
"""
from __future__ import annotations

from enum import Enum, auto

import numpy as np


class WatchdogAction(Enum):
    OK = auto()       # 正常 → 传递命令
    DECAY = auto()    # 短暂丢失 → 速度衰减 (hold with decay)
    STOP = auto()     # 持续丢失/跳变 → 停止 (命令清零)
    ESTOP = auto()    # 严重丢失 → 请求 e_stop


class VisionWatchdog:
    """视觉信号分级: 手丢失/低置信/深度无效/腕跳变."""

    def __init__(self, conf_threshold: float = 0.5,
                 depth_invalid_hold_s: float = 0.2,
                 wrist_jump_mm: float = 150.0,
                 loss_stop_s: float = 0.4,
                 estop_s: float = 1.0,
                 decay_rate: float = 0.5):
        self.conf_threshold = conf_threshold
        self.depth_invalid_hold_s = depth_invalid_hold_s
        self.wrist_jump_mm = wrist_jump_mm
        self.loss_stop_s = loss_stop_s
        self.estop_s = estop_s
        self.decay_rate = decay_rate
        self._loss_start: float | None = None
        self._last_wrist: np.ndarray | None = None

    def update(self, *, hand_present: bool, hand_confidence: float,
               depth_valid: bool, wrist_mm, now: float) -> tuple[WatchdogAction, float]:
        """每帧调用. Returns (action, velocity_scale), scale∈[0,1]. 支持手部重现时自动复位恢复 (Auto-Recovery)."""
        vision_ok = (hand_present and hand_confidence >= self.conf_threshold
                     and depth_valid)
        if vision_ok:
            # 腕跳变 → 立即 STOP 保护 1 帧，并同步更新腕点，下帧自动恢复平滑跟随
            if wrist_mm is not None and self._last_wrist is not None:
                if float(np.linalg.norm(np.asarray(wrist_mm) - self._last_wrist)) > self.wrist_jump_mm:
                    self._last_wrist = np.asarray(wrist_mm)
                    self._loss_start = None
                    return WatchdogAction.STOP, 0.0

            self._last_wrist = np.asarray(wrist_mm) if wrist_mm is not None else None
            self._loss_start = None
            return WatchdogAction.OK, 1.0

        # 视觉丢失: 记录丢失起点, 按时长分级
        if self._loss_start is None:
            self._loss_start = now
        loss_s = now - self._loss_start
        # 深度无效的短暂期 (< hold_s) 只衰减不升级 (P2-⑧)
        if not depth_valid and loss_s < self.depth_invalid_hold_s:
            scale = max(0.0, 1.0 - self.decay_rate * loss_s)
            return WatchdogAction.DECAY, scale
        if loss_s >= self.estop_s:
            return WatchdogAction.ESTOP, 0.0
        if loss_s >= self.loss_stop_s:
            return WatchdogAction.STOP, 0.0
        scale = max(0.0, 1.0 - self.decay_rate * loss_s)
        return WatchdogAction.DECAY, scale

    def reset(self) -> None:
        """重置看门狗丢失状态与历史记录 (离合重置/手动恢复时调用)."""
        self._loss_start = None
        self._last_wrist = None

    check = update

