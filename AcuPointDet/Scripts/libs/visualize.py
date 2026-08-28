# -*- coding: utf-8 -*-
"""结果可视化模块：绘制检测框、37 个穴位关键点、骨架连线、穴位名称。

颜色约定（BGR）：
  检测框  : 绿色
  左侧穴位: 红色
  右侧穴位: 蓝色
  中线穴位: 绿色
  骨架连线: 按两端点类别着色（左红/右蓝/中绿/混合灰）
中文名用 PIL 绘制（cv2.putText 不支持中文）。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from acupoints import (
    ACUPOINT_NAMES_CN, ACUPOINT_SKELETON,
    LEFT_INDICES, RIGHT_INDICES, CENTER_INDICES,
)

DEFAULT_KPT_RADIUS = 6
DEFAULT_BOX_THICK = 3
DEFAULT_LINE_THICK = 2
DEFAULT_FONT_SCALE = 0.5
DEFAULT_FONT_THICK = 1

COLOR_BOX = (0, 255, 0)
COLOR_LEFT = (0, 0, 255)
COLOR_RIGHT = (255, 0, 0)
COLOR_CENTER = (0, 255, 0)
COLOR_SKELETON_MIXED = (180, 180, 180)

FONT_PATH = "/usr/share/fonts/truetype/arphic/uming.ttc"
_font_cache = {}


def _get_font(size):
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def color_for(idx):
    if idx in LEFT_INDICES:
        return COLOR_LEFT
    if idx in RIGHT_INDICES:
        return COLOR_RIGHT
    return COLOR_CENTER


def _skeleton_color(i, j):
    if i in LEFT_INDICES and j in LEFT_INDICES:
        return COLOR_LEFT
    if i in RIGHT_INDICES and j in RIGHT_INDICES:
        return COLOR_RIGHT
    if i in CENTER_INDICES or j in CENTER_INDICES:
        return COLOR_CENTER
    return COLOR_SKELETON_MIXED


def draw_box(img, box, color=COLOR_BOX, thick=DEFAULT_BOX_THICK, label=None):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      DEFAULT_FONT_SCALE, DEFAULT_FONT_THICK)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, DEFAULT_FONT_SCALE,
                    (0, 0, 0), DEFAULT_FONT_THICK, cv2.LINE_AA)


def draw_keypoints_cv(img, kpts, radius=DEFAULT_KPT_RADIUS,
                      draw_skeleton=True, conf_thr=0.3):
    """用 cv2 绘制关键点圆圈与骨架连线（不含中文名）。"""
    n = kpts.shape[0]
    valid_mask = (kpts[:, 2] >= conf_thr) & (kpts[:, 0] >= 0)

    if draw_skeleton:
        for i, j in ACUPOINT_SKELETON:
            if i >= n or j >= n:
                continue
            if not (valid_mask[i] and valid_mask[j]):
                continue
            pi = (int(kpts[i, 0]), int(kpts[i, 1]))
            pj = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(img, pi, pj, _skeleton_color(i, j),
                     DEFAULT_LINE_THICK, cv2.LINE_AA)

    for idx in range(n):
        if not valid_mask[idx]:
            continue
        x, y = int(kpts[idx, 0]), int(kpts[idx, 1])
        color = color_for(idx)
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)


def draw_names_pil(img, kpts, conf_thr=0.3, font_size=18):
    """用 PIL 在图上绘制中文穴位名。"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    n = kpts.shape[0]
    for idx in range(n):
        x, y, c = float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])
        if c < conf_thr or x < 0:
            continue
        bgr = color_for(idx)
        rgb = (bgr[2], bgr[1], bgr[0])
        name = ACUPOINT_NAMES_CN[idx]
        tx, ty = int(x) + 8, int(y) - font_size // 2
        draw.text((tx, ty), name, fill=rgb, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_results(img_bgr, boxes, kpts=None, conf_thr=0.3,
                 draw_names=True, draw_skeleton=True):
    """在 img_bgr 上绘制检测结果（检测框 + 关键点 + 骨架 + 中文名）。

    Args:
        img_bgr: 原图（会被复制，不修改原图）
        boxes: (N, 5) [x1,y1,x2,y2,score] 或 None
        kpts: (37, 3) [x,y,conf] 或 None
    Returns:
        标注后的图像（BGR）
    """
    canvas = img_bgr.copy()
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            label = f"back {b[4]:.2f}"
            draw_box(canvas, b, label=label)
    if kpts is not None:
        draw_keypoints_cv(canvas, kpts, conf_thr=conf_thr,
                          draw_skeleton=draw_skeleton)
        if draw_names:
            canvas = draw_names_pil(canvas, kpts, conf_thr=conf_thr)
    return canvas


def put_info(img, text_lines, origin=(10, 20), color=(255, 255, 255),
             font_scale=0.5, thick=1, bg=True):
    """在左上角写多行信息。"""
    y = origin[1]
    for line in text_lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, thick)
        if bg:
            cv2.rectangle(img, (origin[0] - 2, y - th - 2),
                          (origin[0] + tw + 2, y + 2), (0, 0, 0), -1)
        cv2.putText(img, line, (origin[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)
        y += th + 6
