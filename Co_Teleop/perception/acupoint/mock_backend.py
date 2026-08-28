"""MockAcupointBackend 仿真穴位检测后端 (用于单元测试、CI、无 NPU 节点与离线仿真)."""

from __future__ import annotations

import time
from typing import Optional
import numpy as np

from .base import AcupointDetectorProtocol
from .observation import AcupointObservation


class MockAcupointBackend(AcupointDetectorProtocol):
    """高保真 37 穴位拓扑仿真检测器."""

    def __init__(self, default_confidence: float = 0.95):
        self.default_confidence = float(default_confidence)
        self._initialized = False

    def initialize(self) -> bool:
        """初始化 Mock 状态."""
        self._initialized = True
        return True

    def detect(
        self,
        rgb_image: np.ndarray,
        timestamp: Optional[float] = None,
        target_acupoint: Optional[str] = None,
    ) -> Optional[AcupointObservation]:
        """生成合规的 37 穴位人体背部仿真关键点."""
        if timestamp is None:
            timestamp = time.monotonic()

        h, w = rgb_image.shape[:2] if rgb_image is not None and rgb_image.ndim >= 2 else (480, 640)

        # 模拟背部检测框 [x1, y1, x2, y2]
        bbox = np.array([w * 0.15, h * 0.10, w * 0.85, h * 0.90], dtype=np.float32)

        # 构造 37 个穴位的解剖学相对坐标 [x, y, conf]
        kpts = np.zeros((37, 3), dtype=np.float32)

        # 0: 大椎 (居中偏上)
        kpts[0] = [w * 0.50, h * 0.20, self.default_confidence]

        # 1, 2: 肩井 (左/右肩)
        kpts[1] = [w * 0.30, h * 0.25, self.default_confidence]
        kpts[2] = [w * 0.70, h * 0.25, self.default_confidence]

        # 3~36: 沿脊柱两侧及背部排布
        y_spine = np.linspace(h * 0.28, h * 0.85, 17)
        for i in range(17):
            idx_l = 3 + i * 2
            idx_r = 4 + i * 2
            if idx_l < 37:
                kpts[idx_l] = [w * 0.38, y_spine[i], self.default_confidence]
            if idx_r < 37:
                kpts[idx_r] = [w * 0.62, y_spine[i], self.default_confidence]

        return AcupointObservation(
            keypoints_2d=kpts,
            bbox=bbox,
            confidence=self.default_confidence,
            timestamp=timestamp,
            target_acupoint=target_acupoint,
            is_valid=True,
        )

    def close(self) -> None:
        """关闭并释放."""
        self._initialized = False
