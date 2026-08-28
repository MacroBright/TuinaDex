"""AscendAcupointBackend 华为昇腾 NPU 硬件加速穴位检测后端 (PyACL + OM 离线模型)."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Optional
import numpy as np

from .base import AcupointDetectorProtocol
from .observation import AcupointObservation

logger = logging.getLogger("AscendAcupointBackend")

# 寻找 AcuPointDet 目录
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ACU_LIB_PATH = _REPO_ROOT / "AcuPointDet" / "Scripts" / "libs"
_MODEL_DIR = _REPO_ROOT / "AcuPointDet" / "Model"

if str(_ACU_LIB_PATH) not in sys.path and _ACU_LIB_PATH.exists():
    sys.path.insert(0, str(_ACU_LIB_PATH))


class AscendAcupointBackend(AcupointDetectorProtocol):
    """基于华为昇腾 Ascend PyACL 的 RTMDet + RTMPose 二阶段穴位检测器."""

    def __init__(
        self,
        det_om_path: Optional[str | Path] = None,
        pose_om_path: Optional[str | Path] = None,
        device_id: int = 0,
        score_thr: float = 0.3,
        kpt_thr: float = 0.3,
    ):
        self.det_om_path = Path(det_om_path or (_MODEL_DIR / "RTMDet_Tiny.om"))
        self.pose_om_path = Path(pose_om_path or (_MODEL_DIR / "RTMPose_s.om"))
        self.device_id = device_id
        self.score_thr = score_thr
        self.kpt_thr = kpt_thr

        self._det_infer = None
        self._pose_infer = None
        self._initialized = False

    def initialize(self) -> bool:
        """加载昇腾 OM 模型并初始化 PyACL 上下文."""
        if self._initialized:
            return True

        if not self.det_om_path.exists():
            raise FileNotFoundError(f"未找到 RTMDet OM 模型文件: {self.det_om_path}")
        if not self.pose_om_path.exists():
            raise FileNotFoundError(f"未找到 RTMPose OM 模型文件: {self.pose_om_path}")

        try:
            from det_infer import DetInfer
            from pose_infer import PoseInfer
        except ImportError as e:
            raise ImportError(
                f"无法导入 Ascend 依赖库 (请确认已执行 source /usr/local/Ascend/.../set_env.sh): {e}"
            ) from e

        try:
            logger.info("正在加载 RTMDet 与 RTMPose 昇腾离线模型 (Device %d)...", self.device_id)
            self._det_infer = DetInfer(str(self.det_om_path), device_id=self.device_id)
            self._pose_infer = PoseInfer(str(self.pose_om_path), device_id=self.device_id)
            self._initialized = True
            logger.info("昇腾 OM 穴位检测模型加载成功！")
            return True
        except Exception as e:
            logger.error("昇腾模型初始化失败: %s", e)
            self.close()
            raise RuntimeError(f"Ascend NPU 初始化失败: {e}") from e

    def detect(
        self,
        rgb_image: np.ndarray,
        timestamp: Optional[float] = None,
        target_acupoint: Optional[str] = None,
    ) -> Optional[AcupointObservation]:
        """对工作区图像执行背部区域检测与 37 穴位估计."""
        if not self._initialized:
            raise RuntimeError("AscendAcupointBackend 未初始化，请先调用 initialize().")

        if timestamp is None:
            timestamp = time.monotonic()

        if rgb_image is None or rgb_image.size == 0:
            return None

        # 1. 背部检测 (RTMDet)
        boxes = self._det_infer.detect_on_original(
            rgb_image, score_thr=self.score_thr, max_det=1
        )
        if boxes is None or len(boxes) == 0:
            # 未检测到人体背部
            return None

        best_box = boxes[0]  # [x1, y1, x2, y2, score]
        bbox = best_box[:4]
        det_score = float(best_box[4])

        # 2. 关键点估计 (RTMPose 37 穴位)
        kpts_orig, M, center, scale = self._pose_infer.estimate(
            rgb_image, bbox, kpt_thr=self.kpt_thr
        )
        # kpts_orig 形状为 (37, 3) [x, y, conf]

        return AcupointObservation(
            keypoints_2d=kpts_orig.astype(np.float32),
            bbox=bbox.astype(np.float32),
            confidence=det_score,
            timestamp=timestamp,
            target_acupoint=target_acupoint,
            is_valid=True,
        )

    def close(self) -> None:
        """释放 PyACL 与模型资源."""
        if self._det_infer is not None:
            try:
                self._det_infer.close()
            except Exception:
                pass
            self._det_infer = None

        if self._pose_infer is not None:
            try:
                self._pose_infer.close()
            except Exception:
                pass
            self._pose_infer = None

        self._initialized = False
