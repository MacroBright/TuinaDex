"""Checkerboard detection and frame-quality checks for stereo calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import CheckerboardConfig, QualityConfig


@dataclass(frozen=True)
class DetectionResult:
    """The detected checkerboard corners and the checks governing frame storage."""

    found: bool
    corners: NDArray[np.float32] | None
    sharpness: float
    saturated_fraction: float
    border_ok: bool
    saveable: bool
    reasons: tuple[str, ...]

    @property
    def corner_count(self) -> int:
        """Return the number of detected checkerboard corners."""
        return 0 if self.corners is None else len(self.corners)


def normalize_frame(frame: NDArray[np.generic], rotation_degrees: int) -> NDArray[np.generic]:
    """Normalize a captured frame to its configured orientation."""
    if rotation_degrees == 0:
        return frame
    if rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"unsupported rotation: {rotation_degrees}")


def make_object_points(board: CheckerboardConfig) -> NDArray[np.float32]:
    """Build row-major, millimetre-scale object points for a planar board."""
    points = np.zeros((board.corner_count, 3), dtype=np.float32)
    coordinates = [
        (column * board.square_size_mm, row * board.square_size_mm)
        for row in range(board.rows)
        for column in range(board.columns)
    ]
    points[:, :2] = np.asarray(coordinates, dtype=np.float32)
    return points


def detect_checkerboard(
    frame: NDArray[np.uint8], board: CheckerboardConfig, quality: QualityConfig
) -> DetectionResult:
    """Detect a calibration checkerboard and report whether the frame is saveable."""
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturated_fraction = float(np.mean(grayscale >= 250))
    found, corners = cv2.findChessboardCornersSB(
        grayscale,
        board.pattern_size,
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    )

    if not found or corners is None or len(corners) != board.corner_count:
        return DetectionResult(
            found=False,
            corners=None,
            sharpness=0.0,
            saturated_fraction=saturated_fraction,
            border_ok=False,
            saveable=False,
            reasons=(f"expected {board.corner_count} corners",),
        )

    coordinates = corners.reshape(-1, 2)
    min_x, min_y = coordinates.min(axis=0)
    max_x, max_y = coordinates.max(axis=0)
    height, width = grayscale.shape
    border_ok = bool(
        min_x >= quality.edge_margin_px
        and min_y >= quality.edge_margin_px
        and max_x <= width - quality.edge_margin_px
        and max_y <= height - quality.edge_margin_px
    )

    roi = grayscale[
        int(np.floor(min_y)) : int(np.ceil(max_y)) + 1,
        int(np.floor(min_x)) : int(np.ceil(max_x)) + 1,
    ]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    reasons: list[str] = []
    if not border_ok:
        reasons.append("checkerboard is too close to an image edge")
    if sharpness < quality.min_laplacian_variance:
        reasons.append("checkerboard is too blurry")
    if saturated_fraction > quality.max_saturated_fraction:
        reasons.append("image is severely overexposed")

    return DetectionResult(
        found=True,
        corners=corners,
        sharpness=sharpness,
        saturated_fraction=saturated_fraction,
        border_ok=border_ok,
        saveable=not reasons,
        reasons=tuple(reasons),
    )
