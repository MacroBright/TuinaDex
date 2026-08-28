"""OpenCVCamera 通用 OpenCV / UVC USB 相机后端."""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

from .base import BaseCamera, CameraConfig


class OpenCVCamera(BaseCamera):
    """基于 OpenCV VideoCapture 的相机后端 (支持真实 USB/V4L2 与虚拟 Mock 模式)."""

    def __init__(self, config: CameraConfig, mock_mode: bool = False):
        super().__init__(config)
        self.mock_mode = mock_mode
        self._cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        if self.mock_mode:
            self._is_connected = True
            return True

        try:
            device = int(self.config.device_id) if str(self.config.device_id).isdigit() else str(self.config.device_id)
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2 if isinstance(device, int) else cv2.CAP_ANY)
            if not self._cap.isOpened():
                # 尝试默认后端
                self._cap = cv2.VideoCapture(device)

            if not self._cap.isOpened():
                self._is_connected = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._is_connected = True
            return True
        except Exception:
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        self.stop_stream()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_connected = False

    def _capture_raw(self) -> Optional[np.ndarray]:
        if self.mock_mode:
            # 生成带时间动态色彩的 Mock RGB 图像
            img = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            t = self._frame_count % 255
            img[:, :, 0] = t
            img[:, :, 1] = 128
            img[:, :, 2] = 255 - t
            return img

        if self._cap is None or not self._cap.isOpened():
            return None

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        # 转换为 RGB 格式
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.config.height or rgb.shape[1] != self.config.width:
            rgb = cv2.resize(rgb, (self.config.width, self.config.height))

        return rgb
