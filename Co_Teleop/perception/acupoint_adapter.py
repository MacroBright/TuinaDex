"""AcuPointAdapter 穴位检测顶层调度适配器 (支持 Ascend PyACL 与 Mock 双后端).

职责：
1. 提供统一的工厂方法 create_acupoint_detector；
2. 屏蔽具体硬件推理细节，无缝对接 MockAcupointBackend 与 AscendAcupointBackend；
3. 将输出统一规范为标准 AcupointObservation。
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

from .acupoint.base import AcupointDetectorProtocol
from .acupoint.mock_backend import MockAcupointBackend
from .acupoint.observation import AcupointObservation
from .acupoint.worker import AcuPointWorker

try:
    from .acupoint.ascend_backend import AscendAcupointBackend
except Exception:
    AscendAcupointBackend = None  # type: ignore

logger = logging.getLogger("AcuPointAdapter")


def create_acupoint_detector(
    backend_type: str = "auto",
    device_id: int = 0,
    mock_mode: bool = False,
) -> AcupointDetectorProtocol:
    """创建穴位检测器实例.

    Args:
        backend_type: "auto", "ascend", "mock"
        device_id: NPU 设备 ID
        mock_mode: 是否强制 Mock 模式
    """
    if mock_mode or backend_type == "mock":
        return MockAcupointBackend()

    if backend_type in ("auto", "ascend"):
        if AscendAcupointBackend is not None:
            try:
                ascend_det = AscendAcupointBackend(device_id=device_id)
                ascend_det.initialize()
                return ascend_det
            except Exception as e:
                logger.warning("Ascend NPU 后端不可用 (%s)，自动切换到 Mock 后端", e)
                if backend_type == "ascend":
                    raise
        return MockAcupointBackend()

    raise ValueError(f"未知穴位检测后端类型: {backend_type}")


class AcuPointAdapter:
    """背部穴位检测统一适配器类 (向后兼容)."""

    def __init__(self, mock_mode: bool = False, device_id: int = 0):
        self.mock_mode = mock_mode
        self.device_id = device_id
        self._detector: Optional[AcupointDetectorProtocol] = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化检测器."""
        try:
            self._detector = create_acupoint_detector(
                backend_type="mock" if self.mock_mode else "auto",
                device_id=self.device_id,
                mock_mode=self.mock_mode,
            )
            self._initialized = self._detector.initialize()
            return self._initialized
        except Exception as e:
            logger.error("AcuPointAdapter 初始化失败: %s", e)
            self._initialized = False
            return False

    def detect(
        self,
        rgb_image: np.ndarray,
        timestamp: Optional[float] = None,
        target_acupoint: Optional[str] = None,
    ) -> Optional[AcupointObservation]:
        """对工作区图像执行穴位检测."""
        if not self._initialized or self._detector is None:
            return None
        return self._detector.detect(rgb_image, timestamp=timestamp, target_acupoint=target_acupoint)

    def close(self) -> None:
        """释放资源."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._initialized = False
