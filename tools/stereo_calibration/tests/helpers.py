"""Deterministic image fixtures for stereo checkerboard tests."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


CANVAS_SIZE = (1280, 960)


def synthetic_checkerboard(
    *,
    origin: tuple[int, int] = (390, 305),
    square_px: int = 50,
    background: int = 127,
    blur_kernel: int | None = None,
) -> NDArray[np.uint8]:
    """Return a 9-by-6-inner-corner BGR checkerboard on a 1280-by-960 canvas."""
    width, height = CANVAS_SIZE
    image = np.full((height, width), background, dtype=np.uint8)
    origin_x, origin_y = origin

    for row in range(7):
        for column in range(10):
            value = 20 if (row + column) % 2 == 0 else 230
            top_left = (origin_x + column * square_px, origin_y + row * square_px)
            bottom_right = (
                origin_x + (column + 1) * square_px,
                origin_y + (row + 1) * square_px,
            )
            cv2.rectangle(image, top_left, bottom_right, value, thickness=-1)

    if blur_kernel is not None:
        image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
