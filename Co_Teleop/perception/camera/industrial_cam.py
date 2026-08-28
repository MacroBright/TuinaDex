"""IndustrialCamera 工作区工业相机后端 (支持 GenICam/SDK/高帧率流与自动回退)."""

from __future__ import annotations

from typing import Optional
import numpy as np

from .base import BaseCamera, CameraConfig
from .opencv_cam import OpenCVCamera


class IndustrialCamera(BaseCamera):
    """推拿工作区工业相机抽象实现 (支持专用 SDK 插件与通用高速 V4L2/OpenCV 回退)."""

    def __init__(self, config: CameraConfig, mock_mode: bool = False):
        super().__init__(config)
        self.mock_mode = mock_mode
        # 默认使用高帧率 OpenCVCamera 作为通用后端实现
        self._backend = OpenCVCamera(config, mock_mode=mock_mode)

    def connect(self) -> bool:
        success = self._backend.connect()
        self._is_connected = success
        return success

    def disconnect(self) -> None:
        self.stop_stream()
        self._backend.disconnect()
        self._is_connected = False

    def _capture_raw(self) -> Optional[np.ndarray]:
        return self._backend._capture_raw()
