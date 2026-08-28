"""BaseCamera 相机基类与配置契约."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
import time
from typing import Optional
import numpy as np

from Co_Teleop.recording.teleop_frame import CameraFrame


@dataclass
class CameraConfig:
    """相机通用配置."""

    camera_id: str = "overhead_cam"       # 唯一相机标识
    role: str = "environment"            # 角色: environment (推拿工作区), teleop_control (手势), robot_observation (手腕)
    device_id: str | int = 0             # 设备编号、序列号或网络 URI
    width: int = 640                     # 目标输出宽度
    height: int = 480                    # 目标输出高度
    fps: int = 30                        # 目标帧率
    auto_reconnect: bool = True          # 掉线自动重连


class BaseCamera(ABC):
    """相机抽象基类 (带非阻塞后台采集守护线程)."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._is_connected = False
        self._is_streaming = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[CameraFrame] = None
        self._frame_count = 0
        self._thread: Optional[threading.Thread] = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @abstractmethod
    def connect(self) -> bool:
        """打开设备句柄并初始化."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """安全释放设备资源."""
        pass

    @abstractmethod
    def _capture_raw(self) -> Optional[np.ndarray]:
        """单次捕获一帧原始 RGB 图像 (H, W, 3) uint8."""
        pass

    def start_stream(self) -> None:
        """启动后台连续取帧线程."""
        if self._is_streaming:
            return
        if not self._is_connected:
            if not self.connect():
                return

        self._is_streaming = True
        self._thread = threading.Thread(target=self._stream_worker, name=f"CamWorker-{self.config.camera_id}", daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        """停止后台取帧线程."""
        self._is_streaming = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def get_latest_frame(self) -> Optional[CameraFrame]:
        """非阻塞获取最新一帧 (线程安全)."""
        with self._lock:
            return self._latest_frame

    def _stream_worker(self) -> None:
        """连续采集工作循环."""
        interval = 1.0 / max(1.0, float(self.config.fps))
        while self._is_streaming:
            t_start = time.monotonic()
            img = self._capture_raw()
            if img is not None:
                self._frame_count += 1
                frame = CameraFrame(
                    image=img,
                    timestamp=t_start,
                    frame_id=self._frame_count,
                    camera_id=self.config.camera_id,
                )
                with self._lock:
                    self._latest_frame = frame

            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
