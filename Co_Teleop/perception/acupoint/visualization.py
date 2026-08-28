"""Acupoint 骨架与关键点 OpenCV 绘制工具 (用于 HUD 与数据集可视化)."""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np

# 37 穴位骨架连线定义
ACUPOINT_SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
    (1, 7), (2, 8),
    (7, 9), (8, 10),
    (9, 11), (10, 12),
    (11, 13), (12, 14),
    (13, 15), (14, 16),
    (15, 17), (16, 18),
    (11, 19), (12, 20),
    (19, 21), (20, 22),
    (21, 23), (22, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (31, 33), (32, 34),
    (33, 35), (34, 36),
    (7, 0), (8, 0),
]


def draw_acupoints_on_image(
    image: np.ndarray,
    keypoints_37x3: Optional[np.ndarray],
    bbox: Optional[np.ndarray] = None,
    confidence_thr: float = 0.3,
    target_idx: Optional[int] = None,
) -> np.ndarray:
    """在图像上绘制 37 穴位关键点、骨架连线与目标高亮."""
    if image is None:
        return image

    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    # 1. 绘制背部检测框
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis_img, "Back Region", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    if keypoints_37x3 is None or keypoints_37x3.shape != (37, 3):
        return vis_img

    # 2. 绘制骨架连线
    for idx1, idx2 in ACUPOINT_SKELETON:
        if idx1 < 37 and idx2 < 37:
            p1 = keypoints_37x3[idx1]
            p2 = keypoints_37x3[idx2]
            if p1[2] >= confidence_thr and p2[2] >= confidence_thr:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(vis_img, pt1, pt2, (200, 200, 200), 1, cv2.LINE_AA)

    # 3. 绘制 37 个穴位圆点
    for i in range(37):
        pt = keypoints_37x3[i]
        if pt[2] >= confidence_thr:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                # 区分左右与中心
                color = (0, 255, 0) if i == 0 else ((255, 100, 0) if i % 2 == 1 else (0, 100, 255))
                radius = 4
                if target_idx is not None and i == target_idx:
                    # 目标穴位大圆圈高亮
                    cv2.circle(vis_img, (x, y), 8, (0, 255, 255), 2, cv2.LINE_AA)
                    radius = 5
                cv2.circle(vis_img, (x, y), radius, color, -1, cv2.LINE_AA)

    return vis_img
