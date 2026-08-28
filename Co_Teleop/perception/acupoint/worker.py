"""AcuPointWorker 穴位感知异步后台轮询工作线程 (非阻塞追最新帧 / Drop Oldest)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional
import numpy as np

from Co_Teleop.adapters.state_poller import LatestStateCache
from .base import AcupointDetectorProtocol
from .observation import AcupointObservation

logger = logging.getLogger("AcuPointWorker")


class AcuPointWorker:
    """异步 37 穴位检测守护线程."""

    def __init__(
        self,
        detector: AcupointDetectorProtocol,
        cache: LatestStateCache,
        frame_source: Optional[Callable[[], Optional[tuple[np.ndarray, float]]]] = None,
        max_rate_hz: float = 30.0,
        target_acupoint: Optional[str] = None,
    ):
        self.detector = detector
        self.cache = cache
        self.frame_source = frame_source
        self.max_rate_hz = float(max_rate_hz)
        self.target_acupoint = target_acupoint

        self._interval = 1.0 / max(1.0, self.max_rate_hz)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 性能与状态指标
        self._last_frame_rgb: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0
        self._lock = threading.Lock()
        self._total_inferences = 0
        self._last_infer_cost_ms = 0.0

    def submit_frame(self, rgb_image: np.ndarray, timestamp: Optional[float] = None) -> None:
        """从主感知流非阻塞提交最新相机帧 (覆盖旧帧)."""
        if timestamp is None:
            timestamp = time.monotonic()
        with self._lock:
            self._last_frame_rgb = rgb_image
            self._last_frame_ts = timestamp

    def set_target_acupoint(self, acupoint_name: Optional[str]) -> None:
        """动态更新当前推拿目标穴位 (如 'dazhui', 'jianjing_L')."""
        self.target_acupoint = acupoint_name

    def start(self) -> None:
        """启动后台推理线程."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="AcuPointWorker", daemon=True)
        self._thread.start()
        logger.info("AcuPointWorker 异步推理线程已启动 (Max %.1f Hz)", self.max_rate_hz)

    def stop(self) -> None:
        """安全停止并回收线程."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        logger.info("AcuPointWorker 线程已安全终止")

    def _run_loop(self) -> None:
        """异步推理主循环."""
        while self._running:
            t0 = time.monotonic()

            rgb = None
            ts = 0.0

            # 1. 获取最新待推理帧
            if self.frame_source is not None:
                src_res = self.frame_source()
                if src_res is not None:
                    rgb, ts = src_res
            else:
                with self._lock:
                    if self._last_frame_rgb is not None:
                        rgb = self._last_frame_rgb
                        ts = self._last_frame_ts
                        self._last_frame_rgb = None  # 消费最新帧

            # 2. 执行检测与推理
            if rgb is not None:
                try:
                    t_infer_start = time.monotonic()
                    obs = self.detector.detect(
                        rgb_image=rgb,
                        timestamp=ts if ts > 0 else t0,
                        target_acupoint=self.target_acupoint,
                    )
                    cost_ms = (time.monotonic() - t_infer_start) * 1000.0
                    self._last_infer_cost_ms = cost_ms

                    if obs is not None:
                        self.cache.update_acupoints(obs)
                        self._total_inferences += 1
                except Exception as e:
                    logger.warning("AcuPointWorker 推理异常: %s", e)

            # 3. 速率控制
            elapsed = time.monotonic() - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
