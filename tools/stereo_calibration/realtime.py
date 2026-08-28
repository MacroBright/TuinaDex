"""Realtime stereo processing shared by the Ubuntu point-cloud viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from .cameras import FramePair


@dataclass(frozen=True)
class CalibrationMaps:
    map_left_x: NDArray[np.float32]
    map_left_y: NDArray[np.float32]
    map_right_x: NDArray[np.float32]
    map_right_y: NDArray[np.float32]
    disparity_to_depth: NDArray[np.float64]

    @property
    def image_size(self) -> tuple[int, int]:
        return self.map_left_x.shape[1], self.map_left_x.shape[0]


@dataclass(frozen=True)
class RealtimeSnapshot:
    left_bgr: NDArray[np.uint8]
    depth_bgr: NDArray[np.uint8]
    disparity_px: NDArray[np.float32]
    points_xyz: NDArray[np.float32]
    colors_rgb: NDArray[np.uint8]
    valid_points: int
    median_depth_mm: float
    timestamp_ns: int


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def load_calibration(path: Path | str, image_size: tuple[int, int]) -> CalibrationMaps:
    """Load and validate rectification maps for exactly one logical image size."""
    width, height = image_size
    expected = (height, width)
    try:
        with np.load(Path(path), allow_pickle=False) as values:
            maps = {
                name: np.asarray(values[name], dtype=np.float32).copy()
                for name in ("map_left_x", "map_left_y", "map_right_x", "map_right_y")
            }
            q = np.asarray(values["disparity_to_depth"], dtype=np.float64).copy()
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"could not load stereo calibration: {path}") from exc

    for name, array in maps.items():
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite array with shape {expected}")
    if q.shape != (4, 4) or not np.isfinite(q).all():
        raise ValueError("disparity_to_depth must be a finite 4x4 matrix")
    return CalibrationMaps(
        **{name: _readonly(array) for name, array in maps.items()},
        disparity_to_depth=_readonly(q),
    )


class RealtimeProcessor:
    """Rectify and convert one stereo pair into display and point-cloud data."""

    def __init__(
        self,
        calibration: CalibrationMaps,
        *,
        min_depth_mm: float = 200.0,
        max_depth_mm: float = 5000.0,
        stride: int = 4,
        matcher_factory: Callable[..., Any] = cv2.StereoSGBM_create,
    ) -> None:
        if not np.isfinite(min_depth_mm) or not np.isfinite(max_depth_mm):
            raise ValueError("depth bounds must be finite")
        if min_depth_mm < 0 or max_depth_mm <= min_depth_mm:
            raise ValueError("depth bounds are invalid")
        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")
        self._calibration = calibration
        self._min_depth_mm = float(min_depth_mm)
        self._max_depth_mm = float(max_depth_mm)
        self._stride = stride
        self._scale = 0.5
        self._min_disparity_small = -128
        self._num_disparities_small = 128
        block_size = 7
        self._matcher = matcher_factory(
            minDisparity=self._min_disparity_small,
            numDisparities=self._num_disparities_small,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=2,
            preFilterCap=31,
            uniquenessRatio=8,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def process(self, pair: FramePair) -> RealtimeSnapshot:
        width, height = self._calibration.image_size
        expected = (height, width, 3)
        if pair.left.shape != expected or pair.right.shape != expected:
            raise ValueError(f"stereo pair must contain two uint8 BGR images shaped {expected}")
        if pair.left.dtype != np.uint8 or pair.right.dtype != np.uint8:
            raise ValueError("stereo pair images must use uint8 pixels")

        left_rectified = cv2.remap(
            pair.left,
            self._calibration.map_left_x,
            self._calibration.map_left_y,
            cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            pair.right,
            self._calibration.map_right_x,
            self._calibration.map_right_y,
            cv2.INTER_LINEAR,
        )
        small_size = (width // 2, height // 2)
        left_small = cv2.resize(left_rectified, small_size, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_rectified, small_size, interpolation=cv2.INTER_AREA)
        left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
        disparity_small = (
            self._matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        )
        if disparity_small.shape != (small_size[1], small_size[0]):
            raise ValueError("stereo matcher returned an unexpected disparity shape")
        disparity = cv2.resize(
            disparity_small, (width, height), interpolation=cv2.INTER_NEAREST
        ) / self._scale

        points = cv2.reprojectImageTo3D(
            disparity, self._calibration.disparity_to_depth
        )
        depth = points[:, :, 2]
        full_min_disparity = self._min_disparity_small / self._scale
        valid = (
            (disparity > full_min_disparity + 2.0)
            & (disparity < -2.0)
            & np.isfinite(points).all(axis=2)
            & (depth > self._min_depth_mm)
            & (depth < self._max_depth_mm)
        )
        sampled = valid.copy()
        rows, columns = np.indices(sampled.shape)
        sampled &= (rows % self._stride == 0) & (columns % self._stride == 0)
        xyz = points[sampled].astype(np.float32)
        rgb = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)[sampled].astype(np.uint8)

        depth_level = np.zeros((height, width), dtype=np.uint8)
        depth_level[valid] = np.clip(
            (depth[valid] - self._min_depth_mm)
            / (self._max_depth_mm - self._min_depth_mm)
            * 255.0,
            0,
            255,
        ).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(255 - depth_level, cv2.COLORMAP_TURBO)
        depth_bgr[~valid] = 0
        valid_count = int(valid.sum())
        median_depth = float(np.median(depth[valid])) if valid_count else float("nan")
        return RealtimeSnapshot(
            left_bgr=_readonly(left_rectified),
            depth_bgr=_readonly(depth_bgr),
            disparity_px=_readonly(disparity.astype(np.float32, copy=False)),
            points_xyz=_readonly(xyz),
            colors_rgb=_readonly(rgb),
            valid_points=valid_count,
            median_depth_mm=median_depth,
            timestamp_ns=monotonic_ns(),
        )
