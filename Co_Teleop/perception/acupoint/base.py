"""AcupointDetectorProtocol 穴位检测器抽象协议与通用接口."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
import numpy as np

from .observation import AcupointObservation


@runtime_checkable
class AcupointDetectorProtocol(Protocol):
    """穴位检测器统一协议接口."""

    def initialize(self) -> bool:
        """加载模型权重与初始化计算图/设备上下文."""
        ...

    def detect(
        self,
        rgb_image: np.ndarray,
        timestamp: Optional[float] = None,
        target_acupoint: Optional[str] = None,
    ) -> Optional[AcupointObservation]:
        """对单帧 RGB 图像执行背部区域检测与 37 穴位关键点估计.

        Args:
            rgb_image: (H, W, 3) uint8 BGR/RGB 图像
            timestamp: 图像捕获时间戳 (time.monotonic())
            target_acupoint: 当前期望对准的目标穴位名称

        Returns:
            AcupointObservation 快照或 None (未检测到背部)
        """
        ...

    def close(self) -> None:
        """释放模型与设备资源."""
        ...
