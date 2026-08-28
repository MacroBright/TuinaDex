# -*- coding: utf-8 -*-
"""图像预处理模块。

依据 InferenceGuide.md：
- RTMDet：输入 720×1280，BGR，NCHW，float32，mean/std 按 BGR 顺序
- RTMPose：输入 256×192，RGB，NCHW，float32，mean/std 按 RGB 顺序
  从检测框裁剪并仿射变换到 256×192（外扩 1.25）

注意：仿射变换采用 MMPose 标准 get_affine_transform（3 点），
与训练时预处理保持一致，避免关键点坐标偏移。
"""

import cv2
import numpy as np

DET_INPUT_HW = (480, 640)    # H, W（新训练模型输入，om 内 AIPP 再转 416）
POSE_INPUT_HW = (256, 192)   # H, W
POSE_OUTPUT_WH = (192, 256)  # cv2.warpAffine 的 dsize 是 (W, H)

DET_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
DET_STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)

POSE_MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32)

POSE_EXPAND = 1.25
POSE_BASE_SIZE = 200


def _to_nchw_normalize(img, mean, std):
    x = img.transpose(2, 0, 1)[None].astype(np.float32)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return np.ascontiguousarray(x)


def preprocess_rtmdet(img_bgr, target_hw=DET_INPUT_HW):
    """RTMDet 预处理。

    Args:
        img_bgr: HxWx3 uint8 BGR 原图
        target_hw: 模型输入 (H, W)，默认 (720, 1280)
    Returns:
        (1, 3, H, W) float32，BGR 通道顺序，已归一化
    """
    th, tw = target_hw
    h, w = img_bgr.shape[:2]
    if (h, w) != (th, tw):
        img_bgr = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    return _to_nchw_normalize(img_bgr, DET_MEAN_BGR, DET_STD_BGR)


def preprocess_rtmdet_raw(img_bgr, target_hw=DET_INPUT_HW):
    """RTMDet 原始图预处理（AIPP 模式）。

    AIPP 版 OM 期望 host 送 **HWC 交错 uint8 原始字节**（布局与归一化全由
    设备端 AIPP 完成）。**严禁 transpose 成 NCHW**，否则 AIPP 逐像素按 HWC
    读数据时通道/空间错乱，产生高分假框（实测 0.63）或完全不出框。

    Args:
        img_bgr: HxWx3 uint8 BGR（尺寸应已 letterbox 到 target_hw）
        target_hw: 模型输入 (H, W)，默认 (480, 640)
    Returns:
        HxWx3 uint8 HWC 交错，连续内存
    """
    th, tw = target_hw
    h, w = img_bgr.shape[:2]
    if (h, w) != (th, tw):
        img_bgr = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img_bgr, dtype=np.uint8)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([src_point[0] * cs - src_point[1] * sn,
                     src_point[0] * sn + src_point[1] * cs], dtype=np.float32)


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=(0.0, 0.0), inv=False):
    """MMPose 标准 3 点仿射变换。output_size 为 (W, H)。"""
    scale_tmp = np.array(scale, dtype=np.float32) * POSE_BASE_SIZE
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size[0], output_size[1]
    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = np.array(center, dtype=np.float32) + scale_tmp * np.array(shift, dtype=np.float32)
    src[1, :] = np.array(center, dtype=np.float32) + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def bbox_to_center_scale(bbox, expand=POSE_EXPAND):
    """bbox [x1,y1,x2,y2] -> (center, scale)，scale 为 2 维向量。"""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    scale = max(w, h) / float(POSE_BASE_SIZE) * expand
    return center, np.array([scale, scale], dtype=np.float32)


def crop_and_affine(img_bgr, bbox, out_wh=POSE_OUTPUT_WH, expand=POSE_EXPAND):
    """从检测框裁剪背部区域并仿射到 (W=192, H=256)。

    Returns:
        input_tensor: (1, 3, 256, 192) float32 RGB 已归一化
        M: 2x3 正向仿射矩阵（原图 -> 裁剪图），用于后处理逆变换
        center, scale: 供调试
        crop: BGR 256×192 裁剪图（uint8）
    """
    center, scale = bbox_to_center_scale(bbox, expand=expand)
    M = get_affine_transform(center, scale, 0.0, out_wh, inv=False)
    crop = cv2.warpAffine(img_bgr, M, out_wh, flags=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = _to_nchw_normalize(crop_rgb, POSE_MEAN_RGB, POSE_STD_RGB)
    return x, M, center, scale, crop


def letterbox_resize(img_bgr, target_hw, pad_value=128):
    """等比例缩放并在四周填充灰色，使图像达到 target_hw。

    当原图尺寸已等于 target_hw 时直接返回副本，不做任何处理。

    Args:
        img_bgr: HxWx3 uint8 BGR
        target_hw: (H, W) 目标尺寸
        pad_value: 填充灰度值（0=黑, 128=灰, 255=白），默认 128
    Returns:
        canvas: target_H x target_W x 3 uint8
        r: 缩放比（原图 -> 缩放图）
        (left, top): 缩放图在画布上的左上角偏移
    """
    th, tw = target_hw
    h, w = img_bgr.shape[:2]
    if (h, w) == (th, tw):
        return img_bgr.copy(), 1.0, (0, 0)
    r = min(th / h, tw / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    rs = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), int(pad_value), dtype=np.uint8)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = rs
    return canvas, r, (left, top)


def inv_letterbox(pts, r, pad_left, pad_top):
    """把 letterbox padded 空间的坐标映射回原图空间。

    Args:
        pts: (N, K) float32，K>=2。
             K>=4 时按 [x1,y1,x2,y2,...] 处理（检测框，第 5 列起为 score 等不变换）；
             K==2 时按 [x, y] 点处理。
        r: letterbox 缩放比
        pad_left, pad_top: 画布偏移
    Returns:
        变换后的 pts（副本）
    """
    pts = np.asarray(pts, dtype=np.float32).copy()
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] >= 4:
        pts[:, 0] = (pts[:, 0] - pad_left) / r
        pts[:, 2] = (pts[:, 2] - pad_left) / r
        pts[:, 1] = (pts[:, 1] - pad_top) / r
        pts[:, 3] = (pts[:, 3] - pad_top) / r
    elif pts.shape[1] >= 2:
        pts[:, 0] = (pts[:, 0] - pad_left) / r
        pts[:, 1] = (pts[:, 1] - pad_top) / r
    return pts