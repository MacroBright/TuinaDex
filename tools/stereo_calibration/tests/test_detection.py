"""Behavior tests for stereo calibration checkerboard detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tools.stereo_calibration.config import CheckerboardConfig, QualityConfig
from tools.stereo_calibration import detection
from tools.stereo_calibration.detection import (
    detect_checkerboard,
    make_object_points,
    normalize_frame,
)
from tools.stereo_calibration.tests.helpers import synthetic_checkerboard


BOARD = CheckerboardConfig(columns=9, rows=6, square_size_mm=35.0)
QUALITY = QualityConfig(
    edge_margin_px=12,
    min_laplacian_variance=60.0,
    max_saturated_fraction=0.45,
)


def test_normalize_frame_rotates_180_degrees() -> None:
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

    assert np.array_equal(
        normalize_frame(frame, 180), cv2.rotate(frame, cv2.ROTATE_180)
    )


def test_normalize_frame_leaves_zero_rotation_unchanged() -> None:
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

    assert normalize_frame(frame, 0) is frame


def test_normalize_frame_rejects_unsupported_rotation() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"^unsupported rotation: 90$"):
        normalize_frame(frame, 90)


def test_make_object_points_uses_row_major_millimetre_grid() -> None:
    points = make_object_points(BOARD)

    assert points.shape == (54, 3)
    assert points.dtype == np.float32
    np.testing.assert_array_equal(points[0], (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(points[1], (35.0, 0.0, 0.0))
    np.testing.assert_array_equal(points[9], (0.0, 35.0, 0.0))


def test_sharp_centered_checkerboard_is_detected_and_saveable() -> None:
    result = detect_checkerboard(synthetic_checkerboard(), BOARD, QUALITY)

    assert result.found
    assert result.corner_count == 54
    assert result.saveable
    assert result.reasons == ()


def test_detected_checkerboard_near_an_edge_is_not_saveable() -> None:
    quality = QualityConfig(
        edge_margin_px=80,
        min_laplacian_variance=60.0,
        max_saturated_fraction=0.45,
    )
    result = detect_checkerboard(
        synthetic_checkerboard(origin=(15, 305)), BOARD, quality
    )

    assert result.found
    assert not result.border_ok
    assert not result.saveable
    assert result.reasons == ("checkerboard is too close to an image edge",)


def test_sufficiently_blurred_checkerboard_is_not_saveable() -> None:
    result = detect_checkerboard(
        synthetic_checkerboard(blur_kernel=31), BOARD, QUALITY
    )

    assert result.found
    assert result.sharpness < QUALITY.min_laplacian_variance
    assert not result.saveable
    assert result.reasons == ("checkerboard is too blurry",)


def test_severely_overexposed_checkerboard_is_not_saveable() -> None:
    result = detect_checkerboard(
        synthetic_checkerboard(background=255), BOARD, QUALITY
    )

    assert result.found
    assert result.saturated_fraction > QUALITY.max_saturated_fraction
    assert not result.saveable
    assert result.reasons == ("image is severely overexposed",)


def test_absent_checkerboard_reports_expected_corner_count() -> None:
    frame = np.full((960, 1280, 3), 127, dtype=np.uint8)
    result = detect_checkerboard(frame, BOARD, QUALITY)

    assert not result.found
    assert result.corner_count == 0
    assert result.reasons == ("expected 54 corners",)


def test_overexposed_blank_frame_reports_detection_and_exposure_rejections() -> None:
    frame = np.full((960, 1280, 3), 255, dtype=np.uint8)
    result = detect_checkerboard(frame, BOARD, QUALITY)

    assert not result.found
    assert not result.saveable
    assert result.reasons == (
        "expected 54 corners",
        "image is severely overexposed",
    )


@pytest.mark.parametrize("maximum", [90.0, 89.9])
def test_checkerboard_at_or_rounding_to_the_right_bottom_margin_fails(
    monkeypatch: pytest.MonkeyPatch, maximum: float
) -> None:
    corners = np.full((54, 1, 2), 20.0, dtype=np.float32)
    corners[-1, 0] = (maximum, maximum)
    monkeypatch.setattr(
        detection.cv2,
        "findChessboardCornersSB",
        lambda *_args: (True, corners),
    )
    quality = QualityConfig(
        edge_margin_px=10,
        min_laplacian_variance=0.0,
        max_saturated_fraction=1.0,
    )

    result = detect_checkerboard(np.full((100, 100, 3), 127, dtype=np.uint8), BOARD, quality)

    assert not result.border_ok
    assert result.reasons == ("checkerboard is too close to an image edge",)


def test_detected_corners_are_float32_copied_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector_corners = np.full((54, 1, 2), 20.0, dtype=np.float64)
    monkeypatch.setattr(
        detection.cv2,
        "findChessboardCornersSB",
        lambda *_args: (True, detector_corners),
    )
    quality = QualityConfig(
        edge_margin_px=10,
        min_laplacian_variance=0.0,
        max_saturated_fraction=1.0,
    )

    result = detect_checkerboard(np.full((100, 100, 3), 127, dtype=np.uint8), BOARD, quality)

    assert result.corners is not None
    assert result.corners is not detector_corners
    assert result.corners.shape == detector_corners.shape
    assert result.corners.dtype == np.float32
    assert not result.corners.flags.writeable
    with pytest.raises(ValueError):
        result.corners[0, 0, 0] = 1.0


def test_out_of_range_corners_produce_empty_roi_blur_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corners = np.full((54, 1, 2), 100.0, dtype=np.float32)
    monkeypatch.setattr(
        detection.cv2,
        "findChessboardCornersSB",
        lambda *_args: (True, corners),
    )
    quality = QualityConfig(
        edge_margin_px=0,
        min_laplacian_variance=1.0,
        max_saturated_fraction=1.0,
    )

    result = detect_checkerboard(np.full((20, 20, 3), 127, dtype=np.uint8), BOARD, quality)

    assert result.sharpness == 0.0
    assert result.reasons == (
        "checkerboard is too close to an image edge",
        "checkerboard is too blurry",
    )
