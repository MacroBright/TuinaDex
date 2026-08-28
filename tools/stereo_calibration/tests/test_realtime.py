from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.stereo_calibration.cameras import FramePair
from tools.stereo_calibration.realtime import RealtimeProcessor, load_calibration


def _write_calibration(path: Path, width: int = 8, height: int = 6) -> Path:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    q = np.array(
        [
            [1.0, 0.0, 0.0, -width / 2],
            [0.0, 1.0, 0.0, -height / 2],
            [0.0, 0.0, 0.0, 1000.0],
            [0.0, 0.0, -0.01, 0.0],
        ],
        dtype=np.float64,
    )
    target = path / "calibration.npz"
    np.savez_compressed(
        target,
        map_left_x=x,
        map_left_y=y,
        map_right_x=x,
        map_right_y=y,
        disparity_to_depth=q,
    )
    return target


class _FakeMatcher:
    def __init__(self, fixed: np.ndarray) -> None:
        self.fixed = fixed

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        assert left.shape == right.shape == self.fixed.shape
        return self.fixed.copy()


def _pair(width: int = 8, height: int = 6) -> FramePair:
    left = np.full((height, width, 3), (10, 20, 30), dtype=np.uint8)
    right = left.copy()
    return FramePair(left, right, 10, 11)


def test_load_calibration_rejects_wrong_q_shape(tmp_path: Path) -> None:
    path = _write_calibration(tmp_path)
    with np.load(path) as values:
        payload = {name: values[name] for name in values.files}
    payload["disparity_to_depth"] = np.eye(3)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="disparity_to_depth"):
        load_calibration(path, (8, 6))


def test_balanced_processor_restores_full_resolution_disparity(tmp_path: Path) -> None:
    calibration = load_calibration(_write_calibration(tmp_path), (8, 6))
    fixed = np.full((3, 4), -80 * 16, dtype=np.int16)
    matcher = _FakeMatcher(fixed)
    processor = RealtimeProcessor(
        calibration,
        min_depth_mm=1.0,
        max_depth_mm=5000.0,
        stride=2,
        matcher_factory=lambda **_kwargs: matcher,
    )

    snapshot = processor.process(_pair())

    assert snapshot.left_bgr.shape == (6, 8, 3)
    assert snapshot.depth_bgr.shape == (6, 8, 3)
    assert snapshot.disparity_px.shape == (6, 8)
    assert np.allclose(snapshot.disparity_px, -160.0)
    assert snapshot.points_xyz.shape == (12, 3)
    assert snapshot.colors_rgb.shape == (12, 3)
    assert np.all(snapshot.colors_rgb == np.array([30, 20, 10], dtype=np.uint8))
    assert snapshot.valid_points == 48
    assert snapshot.median_depth_mm == pytest.approx(625.0)
