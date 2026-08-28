# -*- coding: utf-8 -*-
"""RTMDet 检测推理模块。

加载 RTMDet_Tiny.om，对输入图像执行背部区域检测。
输出 (1, 1000, 6) [x1,y1,x2,y2,score,label]，原图坐标，需自行做分数过滤 + NMS。
"""

import logging
import numpy as np

from acl_utils import OmModel
from preprocess import preprocess_rtmdet_raw, letterbox_resize, DET_INPUT_HW
from postprocess import postprocess_rtmdet, DEFAULT_SCORE_THR, DEFAULT_IOU_THR

logger = logging.getLogger("det_infer")

DET_OUTPUT_SHAPE = (1, 1000, 6)


class DetInfer:
    def __init__(self, om_path, device_id=0):
        self.model = OmModel(om_path, device_id=device_id)
        self.input_hw = DET_INPUT_HW

    def detect(self, img_bgr, score_thr=DEFAULT_SCORE_THR,
               iou_thr=DEFAULT_IOU_THR, max_det=1):
        """对一张图执行检测。

        Args:
            img_bgr: HxWx3 uint8 BGR
            score_thr / iou_thr / max_det: 后处理参数
        Returns:
            (N, 5) [x1,y1,x2,y2,score]，坐标基于【模型输入】空间
            （模型输入 480×640，AIPP 内部再转 416；输出坐标在该模型输入空间，
            需映射回原图时用 inv_letterbox / detect_on_original）
        """
        h, w = img_bgr.shape[:2]
        inp = preprocess_rtmdet_raw(img_bgr, target_hw=self.input_hw)
        outs = self.model.infer([inp])
        dets = np.asarray(outs[0]).reshape(DET_OUTPUT_SHAPE)[0]   # (1000, 6)
        boxes = postprocess_rtmdet(dets, score_thr=score_thr, iou_thr=iou_thr,
                                   max_det=max_det, img_hw=self.input_hw)
        return boxes

    def detect_on_original(self, img_bgr, score_thr=DEFAULT_SCORE_THR,
                           iou_thr=DEFAULT_IOU_THR, max_det=1):
        """检测并把框坐标从模型输入空间映射回原图空间。

        预处理采用 letterbox（等比缩放 + 128 灰色填充）到模型输入，
        因此框坐标按 (原图 - pad) / 缩放比 逆映射回原图。
        """
        h, w = img_bgr.shape[:2]
        th, tw = self.input_hw
        img_padded, r, (pad_left, pad_top) = letterbox_resize(
            img_bgr, (th, tw), pad_value=128)
        inp = preprocess_rtmdet_raw(img_padded, target_hw=(th, tw))
        outs = self.model.infer([inp])
        dets = np.asarray(outs[0]).reshape(DET_OUTPUT_SHAPE)[0]
        boxes = postprocess_rtmdet(dets, score_thr=score_thr, iou_thr=iou_thr,
                                   max_det=max_det, img_hw=(th, tw))
        if boxes.shape[0] == 0:
            return boxes
        boxes[:, 0] = (boxes[:, 0] - pad_left) / r
        boxes[:, 2] = (boxes[:, 2] - pad_left) / r
        boxes[:, 1] = (boxes[:, 1] - pad_top) / r
        boxes[:, 3] = (boxes[:, 3] - pad_top) / r
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    def close(self):
        self.model.close()