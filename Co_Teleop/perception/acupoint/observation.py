"""AcupointObservation 穴位观测数据契约.

包含：
- 37 穴位 2D 坐标与置信度 (37, 3) [x, y, confidence]
- 人体背部检测框 bbox (4,) [x1, y1, x2, y2]
- 整体置信度、时间戳与有效性标志
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class AcupointObservation:
    """标准 37 穴位观测数据结构 (不可变)."""

    keypoints_2d: np.ndarray        # (37, 3) 浮点数组 [x_px, y_px, score]
    bbox: np.ndarray                # (4,) 浮点数组 [x1, y1, x2, y2] 背部边界框
    confidence: float               # 背部检测整体置信度
    timestamp: float                # 单调时钟时间戳 (time.monotonic())
    target_acupoint: Optional[str] = None  # 当前推拿目标穴位 (如 "dazhui")
    is_valid: bool = True           # 是否有效 (未发生严重遮挡或模型丢失)

    def get_point(self, index: int) -> np.ndarray:
        """获取指定索引穴位的 [x, y, conf]."""
        return self.keypoints_2d[index]
