"""Deterministic image fixtures for stereo checkerboard tests."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from tools.stereo_calibration.calibration import StereoObservation
from tools.stereo_calibration.config import CheckerboardConfig
from tools.stereo_calibration.detection import make_object_points


CANVAS_SIZE = (1280, 960)

SYNTHETIC_CAMERA_MATRIX = np.array(
    [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
SYNTHETIC_DISTORTION = np.zeros(5, dtype=np.float64)
SYNTHETIC_ROTATION_LEFT_TO_RIGHT = np.eye(3, dtype=np.float64)
SYNTHETIC_TRANSLATION_LEFT_TO_RIGHT = np.array(
    [[-200.0], [0.0], [0.0]], dtype=np.float64
)
SYNTHETIC_BOARD = CheckerboardConfig(columns=9, rows=6, square_size_mm=35.0)


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


def synthetic_stereo_observations() -> list[StereoObservation]:
    """Return exactly 20 deterministic, noise-free stereo board observations."""
    object_points = make_object_points(SYNTHETIC_BOARD)
    observations: list[StereoObservation] = []
    depths = np.linspace(600.0, 950.0, 20)

    for index, depth in enumerate(depths):
        # Diverse, fixed views avoid the planar-calibration degeneracies of merely
        # translating a fronto-parallel board.
        row = index // 5
        column = index % 5
        rvec_left = np.deg2rad(
            np.array(
                [
                    -12.0 + 8.0 * row,
                    -15.0 + 7.5 * column,
                    -6.0 + 4.0 * ((index * 3) % 4),
                ],
                dtype=np.float64,
            )
        ).reshape(3, 1)
        rotation_left_board, _ = cv2.Rodrigues(rvec_left)
        translation_left_board = np.array(
            [
                [45.0 + 12.0 * ((index * 2) % 5)],
                [-105.0 + 13.0 * (index % 4)],
                [float(depth)],
            ],
            dtype=np.float64,
        )
        rotation_right_board = (
            SYNTHETIC_ROTATION_LEFT_TO_RIGHT @ rotation_left_board
        )
        translation_right_board = (
            SYNTHETIC_ROTATION_LEFT_TO_RIGHT @ translation_left_board
            + SYNTHETIC_TRANSLATION_LEFT_TO_RIGHT
        )
        rvec_right, _ = cv2.Rodrigues(rotation_right_board)

        left_corners, _ = cv2.projectPoints(
            object_points,
            rvec_left,
            translation_left_board,
            SYNTHETIC_CAMERA_MATRIX,
            SYNTHETIC_DISTORTION,
        )
        right_corners, _ = cv2.projectPoints(
            object_points,
            rvec_right,
            translation_right_board,
            SYNTHETIC_CAMERA_MATRIX,
            SYNTHETIC_DISTORTION,
        )
        observations.append(
            StereoObservation(
                pair_id=index,
                object_points=np.array(object_points, dtype=np.float32, copy=True),
                left_corners=np.array(left_corners, dtype=np.float32, copy=True),
                right_corners=np.array(right_corners, dtype=np.float32, copy=True),
            )
        )

    return observations
