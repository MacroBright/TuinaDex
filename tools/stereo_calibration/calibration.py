"""Stereo calibration mathematics, validation, and durable result artifacts."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import shutil
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import AppConfig, CheckerboardConfig
from .detection import make_object_points
from .session import SavedPair, SessionStore


_RESULT_LOCK_FILENAME = ".calibration.lock"
_RESULT_THREAD_LOCKS: dict[str, threading.RLock] = {}
_RESULT_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class StereoObservation:
    """Corresponding checkerboard geometry for one saved stereo pair."""

    pair_id: int
    object_points: NDArray[np.float32]
    left_corners: NDArray[np.float32]
    right_corners: NDArray[np.float32]


@dataclass(frozen=True)
class CalibrationResult:
    """All calibrated matrices, remaps, and measured quality indicators."""

    pair_count: int
    left_rms: float
    right_rms: float
    stereo_rms: float
    left_camera_matrix: NDArray[np.float64]
    right_camera_matrix: NDArray[np.float64]
    left_distortion: NDArray[np.float64]
    right_distortion: NDArray[np.float64]
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    essential: NDArray[np.float64]
    fundamental: NDArray[np.float64]
    rectification_left: NDArray[np.float64]
    rectification_right: NDArray[np.float64]
    projection_left: NDArray[np.float64]
    projection_right: NDArray[np.float64]
    disparity_to_depth: NDArray[np.float64]
    map_left_x: NDArray[np.float32]
    map_left_y: NDArray[np.float32]
    map_right_x: NDArray[np.float32]
    map_right_y: NDArray[np.float32]
    per_pair_errors: dict[int, dict[str, float]]
    vertical_error_median_px: float
    vertical_error_p95_px: float

    @property
    def baseline_mm(self) -> float:
        """Return the Euclidean camera-center separation in millimetres."""
        return float(np.linalg.norm(self.translation))


def calibrate_observations(
    observations: Sequence[StereoObservation],
    image_size: tuple[int, int],
    minimum_pairs: int = 18,
) -> CalibrationResult:
    """Calibrate a stereo pair from already matched checkerboard observations."""
    width, height = _validate_image_size(image_size)
    if type(minimum_pairs) is not int or minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be a positive integer")
    samples = list(observations)
    if len(samples) < minimum_pairs:
        raise ValueError(f"at least {minimum_pairs} valid pairs are required")
    _validate_observations(samples)

    object_points = [sample.object_points for sample in samples]
    left_points = [sample.left_corners for sample in samples]
    right_points = [sample.right_corners for sample in samples]
    calibrated_size = (width, height)

    try:
        left_rms, left_matrix, left_distortion, left_rvecs, left_tvecs = cv2.calibrateCamera(
            object_points, left_points, calibrated_size, None, None
        )
        right_rms, right_matrix, right_distortion, right_rvecs, right_tvecs = (
            cv2.calibrateCamera(
                object_points, right_points, calibrated_size, None, None
            )
        )

        per_pair_errors: dict[int, dict[str, float]] = {}
        for index, sample in enumerate(samples):
            left_error = _reprojection_rmse(
                sample.object_points,
                sample.left_corners,
                left_rvecs[index],
                left_tvecs[index],
                left_matrix,
                left_distortion,
            )
            right_error = _reprojection_rmse(
                sample.object_points,
                sample.right_corners,
                right_rvecs[index],
                right_tvecs[index],
                right_matrix,
                right_distortion,
            )
            per_pair_errors[sample.pair_id] = {
                "left_rmse_px": left_error,
                "right_rmse_px": right_error,
                "combined_rmse_px": float(
                    math.sqrt((left_error * left_error + right_error * right_error) / 2.0)
                ),
            }

        (
            stereo_rms,
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            rotation,
            translation,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            object_points,
            left_points,
            right_points,
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            calibrated_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        (
            rectification_left,
            rectification_right,
            projection_left,
            projection_right,
            disparity_to_depth,
            _left_roi,
            _right_roi,
        ) = cv2.stereoRectify(
            left_matrix,
            left_distortion,
            right_matrix,
            right_distortion,
            calibrated_size,
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            left_matrix,
            left_distortion,
            rectification_left,
            projection_left,
            calibrated_size,
            cv2.CV_32FC1,
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            right_matrix,
            right_distortion,
            rectification_right,
            projection_right,
            calibrated_size,
            cv2.CV_32FC1,
        )

        vertical_errors: list[NDArray[np.float32]] = []
        for sample in samples:
            left_rectified = cv2.undistortPoints(
                sample.left_corners,
                left_matrix,
                left_distortion,
                R=rectification_left,
                P=projection_left,
            )
            right_rectified = cv2.undistortPoints(
                sample.right_corners,
                right_matrix,
                right_distortion,
                R=rectification_right,
                P=projection_right,
            )
            vertical_errors.append(
                np.abs(left_rectified[:, 0, 1] - right_rectified[:, 0, 1])
            )
    except cv2.error as exc:
        raise ValueError(f"OpenCV stereo calibration failed: {exc}") from exc

    all_vertical_errors = np.concatenate(vertical_errors).astype(np.float64, copy=False)
    return CalibrationResult(
        pair_count=len(samples),
        left_rms=float(left_rms),
        right_rms=float(right_rms),
        stereo_rms=float(stereo_rms),
        left_camera_matrix=np.asarray(left_matrix, dtype=np.float64),
        right_camera_matrix=np.asarray(right_matrix, dtype=np.float64),
        left_distortion=np.asarray(left_distortion, dtype=np.float64),
        right_distortion=np.asarray(right_distortion, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        essential=np.asarray(essential, dtype=np.float64),
        fundamental=np.asarray(fundamental, dtype=np.float64),
        rectification_left=np.asarray(rectification_left, dtype=np.float64),
        rectification_right=np.asarray(rectification_right, dtype=np.float64),
        projection_left=np.asarray(projection_left, dtype=np.float64),
        projection_right=np.asarray(projection_right, dtype=np.float64),
        disparity_to_depth=np.asarray(disparity_to_depth, dtype=np.float64),
        map_left_x=np.asarray(map_left_x, dtype=np.float32),
        map_left_y=np.asarray(map_left_y, dtype=np.float32),
        map_right_x=np.asarray(map_right_x, dtype=np.float32),
        map_right_y=np.asarray(map_right_y, dtype=np.float32),
        per_pair_errors=per_pair_errors,
        vertical_error_median_px=float(np.median(all_vertical_errors)),
        vertical_error_p95_px=float(np.percentile(all_vertical_errors, 95)),
    )


def load_session_observations(
    session: SessionStore,
    board: CheckerboardConfig,
) -> tuple[list[StereoObservation], list[dict[str, str | int]]]:
    """Return valid observations plus explicit, pair-specific skip diagnostics."""
    observations: list[StereoObservation] = []
    diagnostics: list[dict[str, str | int]] = []
    session_image_size: tuple[int, int] | None = None
    for saved in sorted(session.active_pairs(), key=lambda pair: pair.pair_id):
        try:
            left_image = _read_saved_image(saved.left_path, "left image")
            right_image = _read_saved_image(saved.right_path, "right image")
            left_size = (left_image.shape[1], left_image.shape[0])
            right_size = (right_image.shape[1], right_image.shape[0])
            if left_size != right_size:
                raise ValueError(
                    "left/right captured image dimensions differ: "
                    f"{left_size[0]}x{left_size[1]} versus "
                    f"{right_size[0]}x{right_size[1]}"
                )

            configured_size = saved.metadata.get("image_size")
            if configured_size is None:
                raise ValueError("image_size metadata is missing")
            expected_size = _validate_metadata_image_size(configured_size)
            if expected_size != left_size:
                raise ValueError(
                    "configured image dimensions "
                    f"{expected_size[0]}x{expected_size[1]} do not match captured "
                    f"dimensions {left_size[0]}x{left_size[1]}"
                )

            left_corners = _metadata_corners(saved, "left_corners", board.corner_count)
            right_corners = _metadata_corners(saved, "right_corners", board.corner_count)
            _require_corners_in_bounds(left_corners, left_size, "left_corners")
            _require_corners_in_bounds(right_corners, right_size, "right_corners")
            if session_image_size is not None and left_size != session_image_size:
                raise ValueError(
                    "captured image dimensions "
                    f"{left_size[0]}x{left_size[1]} differ from session dimensions "
                    f"{session_image_size[0]}x{session_image_size[1]}"
                )
            observations.append(
                StereoObservation(
                    pair_id=saved.pair_id,
                    object_points=make_object_points(board).copy(),
                    left_corners=left_corners.reshape(-1, 1, 2),
                    right_corners=right_corners.reshape(-1, 1, 2),
                )
            )
            if session_image_size is None:
                session_image_size = left_size
        except (OSError, TypeError, ValueError) as exc:
            diagnostics.append({"pair_id": saved.pair_id, "reason": str(exc)})
    return observations, diagnostics


def write_calibration_run(
    session: SessionStore,
    result: CalibrationResult,
    config: AppConfig,
    source_pairs: list[SavedPair],
    *,
    skip_diagnostics: Sequence[Mapping[str, str | int]] = (),
) -> Path:
    """Atomically publish a never-overwritten timestamped calibration run."""
    _validate_calibration_result(result, config)
    _validate_source_pairs(result, source_pairs)
    skipped_pairs = _normalize_skip_diagnostics(
        skip_diagnostics, {pair.pair_id for pair in source_pairs}
    )
    readable_sources = _load_preview_sources(source_pairs, config.logical_image_size)
    if len(readable_sources) < 3:
        raise ValueError("at least 3 readable source pairs are required for previews")

    with _result_write_lock(session.results_dir):
        final_dir = _next_result_directory(session.results_dir)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{final_dir.name}-", suffix=".tmp", dir=session.results_dir
            )
        )
        try:
            numeric = _numeric_artifacts(result)
            np.savez_compressed(staging_dir / "stereo_calibration.npz", **numeric)
            _write_yaml_artifact(
                staging_dir / "stereo_calibration.yaml",
                result,
                config.logical_image_size,
                [pair.pair_id for pair in source_pairs],
            )
            report = _build_report(result, config, source_pairs, skipped_pairs)
            (staging_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (staging_dir / "report.md").write_text(
                _report_markdown(report), encoding="utf-8"
            )
            _write_rectified_previews(staging_dir, result, readable_sources[:3])
            _verify_artifacts(staging_dir, result, report, config.logical_image_size)
            _fsync_artifacts(staging_dir)
            if _lstat_or_none(final_dir) is not None:
                raise FileExistsError(f"calibration run already exists: {final_dir.name}")
            os.rename(staging_dir, final_dir)
            _fsync_directory(session.results_dir)
            return final_dir
        except BaseException as primary_error:
            try:
                if _lstat_or_none(staging_dir) is not None:
                    _cleanup_staging(staging_dir)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "calibration artifact operation failed: "
                    f"{primary_error}; staging cleanup failed: {cleanup_error}"
                ) from primary_error
            raise


def _validate_image_size(image_size: object) -> tuple[int, int]:
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must contain width and height")
    width, height = image_size
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image_size must contain a positive integer width and height")
    return width, height


def _validate_observations(samples: list[StereoObservation]) -> None:
    pair_ids: set[int] = set()
    expected_point_count: int | None = None
    reference_object_points: NDArray[np.float32] | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, StereoObservation):
            raise ValueError(f"observation {index} must be a StereoObservation")
        if type(sample.pair_id) is not int or sample.pair_id < 0:
            raise ValueError(f"observation {index} pair ID must be a non-negative integer")
        if sample.pair_id in pair_ids:
            raise ValueError("observation pair IDs must be unique")
        pair_ids.add(sample.pair_id)

        for label, values in (
            ("object_points", sample.object_points),
            ("left_corners", sample.left_corners),
            ("right_corners", sample.right_corners),
        ):
            if not isinstance(values, np.ndarray):
                raise ValueError(f"pair {sample.pair_id} {label} must be a numpy array")
            if values.dtype != np.float32:
                raise ValueError(f"pair {sample.pair_id} {label} must have dtype float32")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"pair {sample.pair_id} {label} must contain only finite values")

        if sample.object_points.ndim != 2 or sample.object_points.shape[1:] != (3,):
            raise ValueError(f"pair {sample.pair_id} object_points must have shape (N, 3)")
        point_count = sample.object_points.shape[0]
        if point_count < 4:
            raise ValueError(f"pair {sample.pair_id} must contain at least 4 points")
        if sample.left_corners.shape != (point_count, 1, 2):
            raise ValueError(
                f"pair {sample.pair_id} left_corners must have shape ({point_count}, 1, 2)"
            )
        if sample.right_corners.shape != (point_count, 1, 2):
            raise ValueError("all observations must have consistent point counts")
        if expected_point_count is None:
            expected_point_count = point_count
            reference_object_points = sample.object_points
        elif point_count != expected_point_count:
            raise ValueError("all observations must have consistent point counts")
        elif not np.array_equal(sample.object_points, reference_object_points):
            raise ValueError("all observations must use identical object points")


def _validate_calibration_result(result: CalibrationResult, config: AppConfig) -> None:
    """Reject malformed calibration data before artifact handling has side effects."""
    if not isinstance(result, CalibrationResult):
        raise ValueError("result must be a CalibrationResult")
    width, height = _validate_image_size(config.logical_image_size)
    reference = config.baseline_reference_mm
    if not _is_plain_finite_number(reference) or reference <= 0:
        raise ValueError("baseline reference must be a positive finite number")
    if type(result.pair_count) is not int or result.pair_count <= 0:
        raise ValueError("pair_count must be a positive plain integer")

    for label, value in (
        ("left_rms", result.left_rms),
        ("right_rms", result.right_rms),
        ("stereo_rms", result.stereo_rms),
        ("vertical_error_median_px", result.vertical_error_median_px),
        ("vertical_error_p95_px", result.vertical_error_p95_px),
    ):
        _require_finite_nonnegative(value, label)

    matrices = {
        "left_camera_matrix": (result.left_camera_matrix, (3, 3)),
        "right_camera_matrix": (result.right_camera_matrix, (3, 3)),
        "rotation": (result.rotation, (3, 3)),
        "essential": (result.essential, (3, 3)),
        "fundamental": (result.fundamental, (3, 3)),
        "rectification_left": (result.rectification_left, (3, 3)),
        "rectification_right": (result.rectification_right, (3, 3)),
        "translation": (result.translation, (3, 1)),
        "projection_left": (result.projection_left, (3, 4)),
        "projection_right": (result.projection_right, (3, 4)),
        "disparity_to_depth": (result.disparity_to_depth, (4, 4)),
    }
    for label, (values, shape) in matrices.items():
        _require_array(values, label, np.dtype(np.float64), shape)
    for label, camera_matrix in (
        ("left_camera_matrix", result.left_camera_matrix),
        ("right_camera_matrix", result.right_camera_matrix),
    ):
        if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise ValueError(f"{label} focal lengths must be positive")

    _require_distortion(result.left_distortion, "left_distortion")
    _require_distortion(result.right_distortion, "right_distortion")
    if not np.allclose(
        result.rotation.T @ result.rotation,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=1e-5,
    ) or not math.isclose(
        float(np.linalg.det(result.rotation)), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError("rotation must be a proper orthonormal rotation matrix")

    expected_map_shape = (height, width)
    for label, values in (
        ("map_left_x", result.map_left_x),
        ("map_left_y", result.map_left_y),
        ("map_right_x", result.map_right_x),
        ("map_right_y", result.map_right_y),
    ):
        _require_array(values, label, np.dtype(np.float32), expected_map_shape)

    if not isinstance(result.per_pair_errors, dict):
        raise ValueError("per_pair_errors must be a dictionary")
    if result.pair_count != len(result.per_pair_errors):
        raise ValueError("pair_count must equal the per-pair error count")
    for pair_id, errors in result.per_pair_errors.items():
        if type(pair_id) is not int or pair_id < 0:
            raise ValueError("per-pair error IDs must be non-negative plain integers")
        if not isinstance(errors, dict):
            raise ValueError(f"pair {pair_id} errors must be a dictionary")
        try:
            left = errors["left_rmse_px"]
            right = errors["right_rmse_px"]
        except KeyError as exc:
            raise ValueError(f"pair {pair_id} errors are missing {exc.args[0]}") from exc
        _require_finite_nonnegative(left, f"pair {pair_id} left_rmse_px")
        _require_finite_nonnegative(right, f"pair {pair_id} right_rmse_px")
        computed = math.sqrt((float(left) ** 2 + float(right) ** 2) / 2.0)
        if "combined_rmse_px" in errors:
            supplied = errors["combined_rmse_px"]
            _require_finite_nonnegative(supplied, f"pair {pair_id} combined_rmse_px")
            if not math.isclose(
                float(supplied), computed, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"pair {pair_id} combined_rmse_px is inconsistent with left/right")


def _validate_source_pairs(
    result: CalibrationResult, source_pairs: Sequence[SavedPair]
) -> None:
    source_ids: list[int] = []
    for index, pair in enumerate(source_pairs):
        if not isinstance(pair, SavedPair):
            raise ValueError(f"source pair {index} must be a SavedPair")
        if type(pair.pair_id) is not int or pair.pair_id < 0:
            raise ValueError("source pair IDs must be non-negative plain integers")
        source_ids.append(pair.pair_id)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source pair IDs must be unique")
    if result.pair_count != len(source_ids):
        raise ValueError("result pair_count must equal the source pair count")
    if set(source_ids) != set(result.per_pair_errors):
        raise ValueError("source pair IDs must exactly match result pair IDs")


def _is_plain_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _require_finite_nonnegative(value: object, label: str) -> None:
    if not _is_plain_finite_number(value):
        raise ValueError(f"{label} must be finite")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")


def _require_array(
    values: object, label: str, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> None:
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{label} must be a numpy array")
    if values.dtype != dtype:
        raise ValueError(f"{label} must have dtype {dtype.name}")
    if values.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")


def _require_distortion(values: object, label: str) -> None:
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{label} must be a numpy array")
    if values.dtype != np.float64:
        raise ValueError(f"{label} must have dtype float64")
    if values.ndim != 2 or 1 not in values.shape or values.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"{label} must be a 2-D row or column vector with 4, 5, 8, 12, or 14 coefficients"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")


def _reprojection_rmse(
    object_points: NDArray[np.float32],
    measured: NDArray[np.float32],
    rotation_vector: NDArray[np.float64],
    translation_vector: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
) -> float:
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    residual = projected.reshape(-1, 2) - measured.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual.astype(np.float64) ** 2, axis=1))))


def _read_saved_image(path: Path, label: str) -> NDArray[np.uint8]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing: {path.name}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{label} is unreadable: {path.name}")
    return image


def _validate_metadata_image_size(value: object) -> tuple[int, int]:
    try:
        return _validate_image_size(value)
    except ValueError as exc:
        raise ValueError("image_size metadata must be [positive width, positive height]") from exc


def _metadata_corners(
    saved: SavedPair, key: str, corner_count: int
) -> NDArray[np.float32]:
    if key not in saved.metadata:
        raise ValueError(f"{key} is missing")
    try:
        corners = np.asarray(saved.metadata[key], dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must contain numeric coordinates") from exc
    if corners.shape != (corner_count, 2):
        raise ValueError(f"{key} must have shape ({corner_count}, 2)")
    if not np.all(np.isfinite(corners)):
        raise ValueError(f"{key} must contain only finite coordinates")
    return np.array(corners, dtype=np.float32, copy=True)


def _require_corners_in_bounds(
    corners: NDArray[np.float32], image_size: tuple[int, int], label: str
) -> None:
    width, height = image_size
    coordinates = corners.reshape(-1, 2)
    if np.any(coordinates[:, 0] < 0) or np.any(coordinates[:, 0] >= width):
        raise ValueError(f"{label} contains coordinates outside image bounds {width}x{height}")
    if np.any(coordinates[:, 1] < 0) or np.any(coordinates[:, 1] >= height):
        raise ValueError(f"{label} contains coordinates outside image bounds {width}x{height}")


def _load_preview_sources(
    source_pairs: Sequence[SavedPair],
    expected_size: tuple[int, int],
) -> list[tuple[SavedPair, NDArray[np.uint8], NDArray[np.uint8]]]:
    readable: list[tuple[SavedPair, NDArray[np.uint8], NDArray[np.uint8]]] = []
    expected_width, expected_height = expected_size
    for pair in source_pairs:
        try:
            left = _read_saved_image(pair.left_path, "left image")
        except ValueError as exc:
            raise ValueError(f"source pair {pair.pair_id} {exc}") from exc
        try:
            right = _read_saved_image(pair.right_path, "right image")
        except ValueError as exc:
            raise ValueError(f"source pair {pair.pair_id} {exc}") from exc
        left_size = (left.shape[1], left.shape[0])
        right_size = (right.shape[1], right.shape[0])
        if left_size != right_size:
            raise ValueError(
                f"source pair {pair.pair_id} left/right image dimensions differ: "
                f"{left_size[0]}x{left_size[1]} versus {right_size[0]}x{right_size[1]}"
            )
        if left_size != expected_size:
            raise ValueError(
                f"source pair {pair.pair_id} image dimensions "
                f"{left_size[0]}x{left_size[1]} do not match configured dimensions "
                f"{expected_width}x{expected_height}"
            )
        readable.append((pair, left, right))
    return readable


def _numeric_artifacts(result: CalibrationResult) -> dict[str, NDArray[Any]]:
    pair_ids = np.array(sorted(result.per_pair_errors), dtype=np.int64)
    pair_errors = np.array(
        [
            [
                result.per_pair_errors[pair_id]["left_rmse_px"],
                result.per_pair_errors[pair_id]["right_rmse_px"],
                _combined_error(result.per_pair_errors[pair_id]),
            ]
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    return {
        "pair_count": np.array(result.pair_count, dtype=np.int64),
        "left_rms": np.array(result.left_rms, dtype=np.float64),
        "right_rms": np.array(result.right_rms, dtype=np.float64),
        "stereo_rms": np.array(result.stereo_rms, dtype=np.float64),
        "left_camera_matrix": result.left_camera_matrix,
        "right_camera_matrix": result.right_camera_matrix,
        "left_distortion": result.left_distortion,
        "right_distortion": result.right_distortion,
        "rotation": result.rotation,
        "translation": result.translation,
        "essential": result.essential,
        "fundamental": result.fundamental,
        "rectification_left": result.rectification_left,
        "rectification_right": result.rectification_right,
        "projection_left": result.projection_left,
        "projection_right": result.projection_right,
        "disparity_to_depth": result.disparity_to_depth,
        "map_left_x": result.map_left_x,
        "map_left_y": result.map_left_y,
        "map_right_x": result.map_right_x,
        "map_right_y": result.map_right_y,
        "pair_ids": pair_ids,
        "per_pair_errors": pair_errors,
        "vertical_error_median_px": np.array(
            result.vertical_error_median_px, dtype=np.float64
        ),
        "vertical_error_p95_px": np.array(
            result.vertical_error_p95_px, dtype=np.float64
        ),
        "baseline_mm": np.array(result.baseline_mm, dtype=np.float64),
    }


def _write_yaml_artifact(
    path: Path,
    result: CalibrationResult,
    image_size: tuple[int, int],
    source_pair_ids: list[int],
) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"could not open YAML artifact for writing: {path}")
    try:
        matrices = {
            "left_camera_matrix": result.left_camera_matrix,
            "right_camera_matrix": result.right_camera_matrix,
            "left_distortion": result.left_distortion,
            "right_distortion": result.right_distortion,
            "rotation": result.rotation,
            "translation": result.translation,
            "essential": result.essential,
            "fundamental": result.fundamental,
            "rectification_left": result.rectification_left,
            "rectification_right": result.rectification_right,
            "projection_left": result.projection_left,
            "projection_right": result.projection_right,
            "disparity_to_depth": result.disparity_to_depth,
        }
        for name, matrix in matrices.items():
            storage.write(name, np.asarray(matrix))
        storage.write("pair_count", int(result.pair_count))
        storage.write("left_rms", float(result.left_rms))
        storage.write("right_rms", float(result.right_rms))
        storage.write("stereo_rms", float(result.stereo_rms))
        storage.write("vertical_error_median_px", float(result.vertical_error_median_px))
        storage.write("vertical_error_p95_px", float(result.vertical_error_p95_px))
        storage.write("baseline_mm", float(result.baseline_mm))
        storage.write("image_width", int(image_size[0]))
        storage.write("image_height", int(image_size[1]))
        storage.write(
            "source_pair_ids", np.asarray(source_pair_ids, dtype=np.int32).reshape(-1, 1)
        )
    finally:
        storage.release()


def _combined_error(errors: dict[str, float]) -> float:
    if "combined_rmse_px" in errors:
        value = errors["combined_rmse_px"]
    else:
        left = errors["left_rmse_px"]
        right = errors["right_rmse_px"]
        value = math.sqrt((left * left + right * right) / 2.0)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("per-pair reprojection errors must be finite and non-negative")
    return float(value)


def _normalize_skip_diagnostics(
    diagnostics: Sequence[Mapping[str, str | int]], source_pair_ids: set[int]
) -> list[dict[str, str | int]]:
    if isinstance(diagnostics, (str, bytes)) or not isinstance(diagnostics, Sequence):
        raise ValueError("skip diagnostics must be a sequence")
    normalized: list[dict[str, str | int]] = []
    seen: set[int] = set()
    for index, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"skip diagnostic {index} must be a mapping")
        if set(diagnostic) != {"pair_id", "reason"}:
            raise ValueError(
                f"skip diagnostic {index} must contain exactly pair_id and reason"
            )
        pair_id = diagnostic["pair_id"]
        reason = diagnostic["reason"]
        if type(pair_id) is not int or pair_id < 0:
            raise ValueError(f"skip diagnostic {index} pair ID must be non-negative")
        if pair_id in seen:
            raise ValueError(f"skip diagnostic pair ID {pair_id} is duplicated")
        if pair_id in source_pair_ids:
            raise ValueError(
                f"skip diagnostic pair ID {pair_id} is also a calibration source"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"skip diagnostic {index} reason must be non-empty text")
        if any(ord(character) < 32 and character != "\t" for character in reason):
            raise ValueError(f"skip diagnostic {index} reason contains control characters")
        seen.add(pair_id)
        normalized.append({"pair_id": pair_id, "reason": reason})
    return sorted(normalized, key=lambda item: int(item["pair_id"]))


def _build_report(
    result: CalibrationResult,
    config: AppConfig,
    source_pairs: Sequence[SavedPair],
    skipped_pairs: Sequence[Mapping[str, str | int]] = (),
) -> dict[str, Any]:
    rms_max = max(result.left_rms, result.right_rms, result.stereo_rms)
    tx, ty, tz = (float(value) for value in result.translation.reshape(3))
    horizontal_baseline = abs(tx) > max(abs(ty), abs(tz))
    if not all(
        math.isfinite(value) and value >= 0
        for value in (
            result.left_rms,
            result.right_rms,
            result.stereo_rms,
            result.vertical_error_median_px,
            result.vertical_error_p95_px,
            result.baseline_mm,
        )
    ):
        raise ValueError("calibration metrics must be finite and non-negative")
    if rms_max > 1.5:
        status = "fail"
    elif rms_max > 1.0:
        status = "warning"
    elif (
        result.vertical_error_median_px >= 1.0
        or result.vertical_error_p95_px >= 2.0
        or not horizontal_baseline
    ):
        status = "warning"
    else:
        status = "pass"

    combined = {
        int(pair_id): _combined_error(errors)
        for pair_id, errors in sorted(result.per_pair_errors.items())
    }
    values = np.asarray(list(combined.values()), dtype=np.float64)
    if values.size:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
    else:
        median = 0.0
        mad = 0.0
    outlier_threshold = max(1.5, median + 3.0 * mad)
    outliers = [
        {"pair_id": pair_id, "combined_reprojection_error_px": error}
        for pair_id, error in combined.items()
        if error > outlier_threshold
    ]

    reference = float(config.baseline_reference_mm)
    tolerance = 0.15 * reference
    baseline_warning = abs(result.baseline_mm - reference) > tolerance
    return {
        "status": status,
        "metrics": {
            "pair_count": result.pair_count,
            "left_rms_px": result.left_rms,
            "right_rms_px": result.right_rms,
            "stereo_rms_px": result.stereo_rms,
            "vertical_error_median_px": result.vertical_error_median_px,
            "vertical_error_p95_px": result.vertical_error_p95_px,
            "baseline_mm": result.baseline_mm,
            "translation_xyz_mm": [tx, ty, tz],
            "horizontal_baseline": horizontal_baseline,
        },
        "thresholds": {
            "rms_pass_max_px": 1.0,
            "rms_fail_above_px": 1.5,
            "vertical_median_warning_at_px": 1.0,
            "vertical_p95_warning_at_px": 2.0,
            "outlier_threshold_px": outlier_threshold,
            "outlier_minimum_px": 1.5,
            "outlier_median_px": median,
            "outlier_mad_px": mad,
            "baseline_tolerance_fraction": 0.15,
        },
        "source_pair_ids": [pair.pair_id for pair in source_pairs],
        "skipped_pairs": [dict(item) for item in skipped_pairs],
        "per_pair_errors": {
            str(pair_id): {
                "left_rmse_px": float(errors["left_rmse_px"]),
                "right_rmse_px": float(errors["right_rmse_px"]),
                "combined_rmse_px": combined[pair_id],
            }
            for pair_id, errors in sorted(result.per_pair_errors.items())
        },
        "suspected_outliers": outliers,
        "baseline_warning": {
            "warning": baseline_warning,
            "measured_mm": result.baseline_mm,
            "reference_mm": reference,
            "allowed_difference_mm": tolerance,
        },
    }


def _report_markdown(report: dict[str, Any]) -> str:
    status_chinese = {"pass": "通过", "warning": "警告", "fail": "失败"}[
        report["status"]
    ]
    metrics = report["metrics"]
    baseline = report["baseline_warning"]
    outliers = report["suspected_outliers"]
    skipped_pairs = report["skipped_pairs"]
    lines = [
        "# 双目相机标定报告",
        "",
        f"标定状态：{status_chinese}",
        f"基线警告：{'是' if baseline['warning'] else '否'}",
        "",
        "## 标定指标",
        "",
        f"- 有效图像组数：{metrics['pair_count']}",
        f"- 左相机 RMS：{metrics['left_rms_px']:.6f} px",
        f"- 右相机 RMS：{metrics['right_rms_px']:.6f} px",
        f"- 双目 RMS：{metrics['stereo_rms_px']:.6f} px",
        f"- 垂直误差中位数：{metrics['vertical_error_median_px']:.6f} px",
        f"- 垂直误差 P95：{metrics['vertical_error_p95_px']:.6f} px",
        f"- 基线长度：{metrics['baseline_mm']:.3f} mm",
        "- 平移向量 X/Y/Z："
        + ", ".join(f"{value:.3f}" for value in metrics["translation_xyz_mm"])
        + " mm",
        f"- 水平基线：{'是' if metrics['horizontal_baseline'] else '否'}",
        f"- 参考基线：{baseline['reference_mm']:.3f} mm",
        "",
        "## 数据与离群检查",
        "",
        "- 源图像组 ID：" + ", ".join(map(str, report["source_pair_ids"])),
    ]
    if skipped_pairs:
        lines.append("- 跳过的图像组：")
        lines.extend(
            f"  - {item['pair_id']}（{item['reason']}）" for item in skipped_pairs
        )
    else:
        lines.append("- 跳过的图像组：无")
    if outliers:
        lines.append("- 疑似离群组：")
        lines.extend(
            "  - "
            f"{item['pair_id']}（组合重投影误差 "
            f"{item['combined_reprojection_error_px']:.3f} px）"
            for item in outliers
        )
    else:
        lines.append("- 疑似离群组：无")
    lines.extend(
        [
            "",
            "## 判定阈值",
            "",
            "- RMS 不大于 1.0 px 为通过，大于 1.0 px 为警告，"
            "大于 1.5 px 为失败。",
            "- 垂直误差中位数达到 1.0 px 或 P95 达到 2.0 px 为警告。",
            "- 基线与参考值偏差超过 15% 时给出基线警告。",
            "- 疑似离群组仅报告，未自动排除，也未删除原图。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_rectified_previews(
    staging_dir: Path,
    result: CalibrationResult,
    sources: Sequence[tuple[SavedPair, NDArray[np.uint8], NDArray[np.uint8]]],
) -> None:
    previews_dir = staging_dir / "previews"
    previews_dir.mkdir()
    for pair, left, right in sources:
        left_rectified = cv2.remap(
            left, result.map_left_x, result.map_left_y, cv2.INTER_LINEAR
        )
        right_rectified = cv2.remap(
            right, result.map_right_x, result.map_right_y, cv2.INTER_LINEAR
        )
        preview = np.hstack((left_rectified, right_rectified))
        for y in range(40, preview.shape[0], 40):
            cv2.line(preview, (0, y), (preview.shape[1] - 1, y), (0, 255, 0), 1)
        path = previews_dir / f"pair_{pair.pair_id:04d}_rectified.png"
        if not cv2.imwrite(str(path), preview):
            raise OSError(f"could not write rectified preview for pair {pair.pair_id}")


def _verify_artifacts(
    staging_dir: Path,
    result: CalibrationResult,
    report: dict[str, Any],
    image_size: tuple[int, int],
) -> None:
    """Reopen every artifact and validate its interoperable structure."""
    expected_numeric = _numeric_artifacts(result)
    npz_path = staging_dir / "stereo_calibration.npz"
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_numeric):
                raise ValueError("NPZ artifact keys are incomplete or unexpected")
            for name, expected in expected_numeric.items():
                loaded = archive[name]
                expected_array = np.asarray(expected)
                if loaded.shape != expected_array.shape:
                    raise ValueError(f"NPZ artifact {name} has an unexpected shape")
                if loaded.dtype != expected_array.dtype:
                    raise ValueError(f"NPZ artifact {name} has an unexpected dtype")
                if not np.all(np.isfinite(loaded)):
                    raise ValueError(f"NPZ artifact {name} contains non-finite values")
                if np.issubdtype(loaded.dtype, np.floating):
                    values_match = np.allclose(
                        loaded, expected_array, rtol=1e-12, atol=1e-12
                    )
                else:
                    values_match = np.array_equal(loaded, expected_array)
                if not values_match:
                    raise ValueError(f"NPZ artifact {name} value does not match expected")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("NPZ artifact"):
            raise
        raise ValueError(f"NPZ artifact failed reopen validation: {exc}") from exc

    yaml_matrices = {
        "left_camera_matrix": result.left_camera_matrix,
        "right_camera_matrix": result.right_camera_matrix,
        "left_distortion": result.left_distortion,
        "right_distortion": result.right_distortion,
        "rotation": result.rotation,
        "translation": result.translation,
        "essential": result.essential,
        "fundamental": result.fundamental,
        "rectification_left": result.rectification_left,
        "rectification_right": result.rectification_right,
        "projection_left": result.projection_left,
        "projection_right": result.projection_right,
        "disparity_to_depth": result.disparity_to_depth,
    }
    storage = cv2.FileStorage(
        str(staging_dir / "stereo_calibration.yaml"), cv2.FILE_STORAGE_READ
    )
    if not storage.isOpened():
        raise ValueError("YAML artifact failed reopen validation")
    try:
        for name, expected in yaml_matrices.items():
            values = storage.getNode(name).mat()
            if values is None or values.shape != expected.shape:
                raise ValueError(f"YAML artifact {name} has an unexpected shape")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"YAML artifact {name} contains non-finite values")
            if not np.allclose(values, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"YAML artifact {name} values do not match expected")

        width, height = image_size
        yaml_scalars = {
            "pair_count": result.pair_count,
            "left_rms": result.left_rms,
            "right_rms": result.right_rms,
            "stereo_rms": result.stereo_rms,
            "vertical_error_median_px": result.vertical_error_median_px,
            "vertical_error_p95_px": result.vertical_error_p95_px,
            "baseline_mm": result.baseline_mm,
            "image_width": width,
            "image_height": height,
        }
        for name, expected in yaml_scalars.items():
            node = storage.getNode(name)
            if node.empty():
                raise ValueError(f"YAML artifact {name} is missing")
            value = float(node.real())
            if not math.isfinite(value):
                raise ValueError(f"YAML artifact {name} is non-finite")
            if not math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"YAML artifact {name} does not match expected")

        expected_source_ids = np.asarray(
            report["source_pair_ids"], dtype=np.int32
        ).reshape(-1, 1)
        source_ids = storage.getNode("source_pair_ids").mat()
        if source_ids is None or source_ids.shape != expected_source_ids.shape:
            raise ValueError("YAML artifact source_pair_ids has an unexpected shape")
        if not np.array_equal(source_ids, expected_source_ids):
            raise ValueError("YAML artifact source_pair_ids does not match expected")
    finally:
        storage.release()

    try:
        reopened_report = json.loads(
            (staging_dir / "report.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON report failed reopen validation: {exc}") from exc
    if reopened_report != report:
        raise ValueError("JSON report changed during reopen validation")
    try:
        markdown = (staging_dir / "report.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Markdown report failed reopen validation: {exc}") from exc
    if markdown != _report_markdown(report):
        raise ValueError("Markdown report does not match expected content")

    previews = sorted((staging_dir / "previews").glob("*.png"))
    expected_preview_names = [
        f"pair_{pair_id:04d}_rectified.png"
        for pair_id in report["source_pair_ids"][:3]
    ]
    if [path.name for path in previews] != expected_preview_names:
        raise ValueError("preview artifact names do not match expected source pairs")
    expected_shape = (height, width * 2, 3)
    for path in previews:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"preview {path.name} failed reopen validation")
        if image.shape != expected_shape:
            raise ValueError(
                f"preview {path.name} has unexpected shape {image.shape}; "
                f"expected {expected_shape}"
            )
        for y in range(40, height, 40):
            if not np.all(image[y, :, :] == np.array([0, 255, 0], dtype=np.uint8)):
                raise ValueError(f"preview {path.name} is missing horizontal line at y={y}")


def _fsync_artifacts(staging_dir: Path) -> None:
    artifact_files = [
        staging_dir / "stereo_calibration.npz",
        staging_dir / "stereo_calibration.yaml",
        staging_dir / "report.json",
        staging_dir / "report.md",
        *sorted((staging_dir / "previews").glob("*.png")),
    ]
    for path in artifact_files:
        _fsync_regular_file(path)
    _fsync_directory(staging_dir / "previews")
    _fsync_directory(staging_dir)


def _fsync_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise OSError(f"refusing to fsync symlink artifact: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise OSError(f"artifact is not a regular file: {path}")
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _fsync_directory(directory: Path) -> None:
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            os.fsync(directory_descriptor)
        except OSError as exc:
            unsupported = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(directory_descriptor)


@contextmanager
def _result_write_lock(results_dir: Path) -> Iterator[None]:
    """Serialize cooperating artifact publishers across threads and processes."""
    state = _lstat_or_none(results_dir)
    if state is None or not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise OSError(f"results directory is missing or invalid: {results_dir}")
    key = str(results_dir.absolute())
    with _RESULT_THREAD_LOCKS_GUARD:
        thread_lock = _RESULT_THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock:
        lock_path = results_dir / _RESULT_LOCK_FILENAME
        lock_state = _lstat_or_none(lock_path)
        if lock_state is not None:
            if stat.S_ISLNK(lock_state.st_mode):
                raise OSError(f"calibration result lock is a symlink: {lock_path}")
            if not stat.S_ISREG(lock_state.st_mode):
                raise OSError(f"calibration result lock is not regular: {lock_path}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_descriptor = os.open(lock_path, flags, 0o600)
        primary_error: BaseException | None = None
        lock_acquired = False
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise OSError(f"calibration result lock is not regular: {lock_path}")
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            lock_acquired = True
            yield
        except BaseException as exc:
            primary_error = exc

        if lock_acquired:
            try:
                _unlock_result_lock(lock_descriptor)
            except BaseException as exc:
                cleanup_errors.append(("unlock", exc))
        try:
            _close_result_lock(lock_descriptor)
        except BaseException as exc:
            cleanup_errors.append(("close", exc))

        if primary_error is not None:
            if cleanup_errors:
                details = "; ".join(
                    f"{operation} failure: {error}"
                    for operation, error in cleanup_errors
                )
                raise RuntimeError(
                    f"calibration result operation failed: {primary_error}; "
                    f"lock cleanup failed: {details}"
                ) from primary_error
            raise primary_error
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0][1]
        if cleanup_errors:
            details = "; ".join(
                f"{operation} failure: {error}" for operation, error in cleanup_errors
            )
            raise RuntimeError(f"calibration result lock cleanup failed: {details}") from (
                cleanup_errors[0][1]
            )


def _unlock_result_lock(lock_descriptor: int) -> None:
    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)


def _close_result_lock(lock_descriptor: int) -> None:
    os.close(lock_descriptor)


def _cleanup_staging(staging_dir: Path) -> None:
    state = _lstat_or_none(staging_dir)
    if state is None:
        return
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise OSError(f"staging path is not a regular directory: {staging_dir}")
    shutil.rmtree(staging_dir)


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _current_datetime() -> datetime:
    return datetime.now()


def _next_result_directory(results_dir: Path) -> Path:
    timestamp = _current_datetime()
    while True:
        candidate = results_dir / timestamp.strftime("%Y%m%d-%H%M%S%f")
        if _lstat_or_none(candidate) is None:
            return candidate
        timestamp += timedelta(microseconds=1)
