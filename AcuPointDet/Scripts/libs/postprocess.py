# -*- coding: utf-8 -*-
"""后处理模块。

- RTMDet：分数阈值过滤 + NMS（模型导出时 --no-nms，输出 1000 候选框）
- RTMPose：关键点坐标从 256×192 裁剪图逆仿射回原图
"""

import cv2
import numpy as np

DEFAULT_SCORE_THR = 0.25
DEFAULT_IOU_THR = 0.65


def nms(boxes, scores, iou_thr):
    """纯 numpy NMS。boxes: (N,4) [x1,y1,x2,y2]，scores: (N,)。返回保留索引。"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def postprocess_rtmdet(dets, score_thr=DEFAULT_SCORE_THR, iou_thr=DEFAULT_IOU_THR,
                       max_det=1, img_hw=None):
    """RTMDet 后处理。

    Args:
        dets: (1000, 6) [x1,y1,x2,y2,score,label]，原图坐标
        score_thr: 分数阈值
        iou_thr: NMS IoU 阈值
        max_det: 最多保留框数（单类背部检测，默认 1）
        img_hw: 可选 (H,W)，用于把框坐标裁剪到图像范围内
    Returns:
        (N, 5) [x1,y1,x2,y2,score]，按分数降序
    """
    dets = np.asarray(dets).reshape(-1, 6)
    valid = dets[:, 4] > score_thr
    dets = dets[valid]
    if dets.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = dets[:, :4]
    scores = dets[:, 4]

    try:
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                score_thr, iou_thr)
        keep = np.asarray(keep).flatten().astype(np.int64)
    except Exception:
        keep = nms(boxes, scores, iou_thr)

    if keep.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = boxes[keep]
    scores = scores[keep]
    order = scores.argsort()[::-1][:max_det]
    boxes = boxes[order]
    scores = scores[order]

    if img_hw is not None:
        ih, iw = img_hw
        boxes[:, 0] = np.clip(boxes[:, 0], 0, iw - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, ih - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, iw - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, ih - 1)

    return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)


def postprocess_rtmpose(kpts, M, kpt_thr=0.3):
    """RTMPose 后处理：关键点从 256×192 裁剪图逆仿射回原图。

    Args:
        kpts: (37, 3) [x, y, conf]，裁剪图坐标
        M: 2x3 正向仿射矩阵（原图 -> 裁剪图）
        kpt_thr: 置信度阈值，低于该值的关键点坐标置为 -1
    Returns:
        (37, 3) [x, y, conf]，原图坐标
    """
    kpts = np.asarray(kpts).reshape(-1, 3).astype(np.float32)
    xy = kpts[:, :2].copy()
    conf = kpts[:, 2].copy()

    M_inv = cv2.invertAffineTransform(M)
    ones = np.ones((xy.shape[0], 1), dtype=np.float32)
    xy_h = np.concatenate([xy, ones], axis=1)
    xy_orig = xy_h @ M_inv.T

    low = conf < kpt_thr
    xy_orig[low] = -1.0

    return np.concatenate([xy_orig, conf[:, None]], axis=1).astype(np.float32)