# -*- coding: utf-8 -*-
"""RTMPose 关键点推理模块。

加载 RTMPose_s.om，在检测框裁剪出的背部区域上估计 37 个穴位关键点。
输出 (1, 37, 3) [x, y, conf]，裁剪图 256×192 坐标，需逆仿射回原图。
"""

import logging
import numpy as np

from acl_utils import OmModel
from preprocess import crop_and_affine, POSE_INPUT_HW, POSE_OUTPUT_WH, POSE_EXPAND
from postprocess import postprocess_rtmpose

logger = logging.getLogger("pose_infer")

POSE_OUTPUT_SHAPE = (1, 37, 3)
DEFAULT_KPT_THR = 0.3


class PoseInfer:
    def __init__(self, om_path, device_id=0):
        self.model = OmModel(om_path, device_id=device_id)
        self.input_hw = POSE_INPUT_HW
        self.out_wh = POSE_OUTPUT_WH

    def estimate(self, img_bgr, bbox, kpt_thr=DEFAULT_KPT_THR, expand=POSE_EXPAND,
                 return_crop=False):
        """对指定 bbox 区域执行关键点估计。

        Args:
            img_bgr: HxWx3 uint8 BGR 原图
            bbox: [x1,y1,x2,y2]，原图坐标
            kpt_thr: 关键点置信度阈值
            expand: bbox 外扩系数（默认 1.25，与训练一致）
            return_crop: 是否返回裁剪图（BGR 256×192）
        Returns:
            kpts: (37, 3) [x, y, conf]，原图坐标（低置信度点坐标置 -1）
            M: 2x3 正向仿射矩阵
            center, scale: 供调试
            crop: (可选) 裁剪图 BGR 256×192
        """
        inp, M, center, scale, crop = crop_and_affine(
            img_bgr, bbox, out_wh=self.out_wh, expand=expand)
        outs = self.model.infer([inp])
        kpts = np.asarray(outs[0]).reshape(POSE_OUTPUT_SHAPE)[0]   # (37, 3)
        kpts_orig = postprocess_rtmpose(kpts, M, kpt_thr=kpt_thr)
        if return_crop:
            return kpts_orig, M, center, scale, crop
        return kpts_orig, M, center, scale

    def close(self):
        self.model.close()