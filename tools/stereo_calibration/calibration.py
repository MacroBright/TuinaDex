"""Stereo calibration mathematics, validation, and durable result artifacts."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import AppConfig, CheckerboardConfig
from .detection import make_object_points
from .session import SavedPair, SessionStore


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
            observations.append(
                StereoObservation(
                    pair_id=saved.pair_id,
                    object_points=make_object_points(board).copy(),
                    left_corners=left_corners.reshape(-1, 1, 2),
                    right_corners=right_corners.reshape(-1, 1, 2),
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            diagnostics.append({"pair_id": saved.pair_id, "reason": str(exc)})
    return observations, diagnostics


def write_calibration_run(
    session: SessionStore,
    result: CalibrationResult,
    config: AppConfig,
    source_pairs: list[SavedPair],
) -> Path:
    """Atomically publish a never-overwritten timestamped calibration run."""
    reference = config.baseline_reference_mm
    if type(reference) not in (int, float) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("baseline reference must be a positive finite number")

    readable_sources = _load_preview_sources(source_pairs)
    if len(readable_sources) < 3:
        raise ValueError("at least 3 readable source pairs are required for previews")

    session.results_dir.mkdir(parents=False, exist_ok=True)
    final_dir = _next_result_directory(session.results_dir)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}-", suffix=".tmp", dir=session.results_dir)
    )
    try:
        numeric = _numeric_artifacts(result)
        np.savez_compressed(staging_dir / "stereo_calibration.npz", **numeric)
        _write_yaml_artifact(
            staging_dir / "stereo_calibration.yaml",
            result,
            config.capture.image_size,
            [pair.pair_id for pair in source_pairs],
        )
        report = _build_report(result, config, source_pairs)
        (staging_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "report.md").write_text(
            _report_markdown(report), encoding="utf-8"
        )
        _write_rectified_previews(staging_dir, result, readable_sources[:3])
        os.rename(staging_dir, final_dir)
        return final_dir
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
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


def _load_preview_sources(
    source_pairs: Sequence[SavedPair],
) -> list[tuple[SavedPair, NDArray[np.uint8], NDArray[np.uint8]]]:
    readable: list[tuple[SavedPair, NDArray[np.uint8], NDArray[np.uint8]]] = []
    seen: set[int] = set()
    for pair in source_pairs:
        if type(pair.pair_id) is not int or pair.pair_id < 0:
            raise ValueError("source pair IDs must be non-negative integers")
        if pair.pair_id in seen:
            raise ValueError("source pair IDs must be unique")
        seen.add(pair.pair_id)
        try:
            left = _read_saved_image(pair.left_path, "left image")
            right = _read_saved_image(pair.right_path, "right image")
        except ValueError:
            continue
        if left.shape[:2] != right.shape[:2]:
            continue
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


def _build_report(
    result: CalibrationResult, config: AppConfig, source_pairs: Sequence[SavedPair]
) -> dict[str, Any]:
    rms_max = max(result.left_rms, result.right_rms, result.stereo_rms)
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
        f"- 参考基线：{baseline['reference_mm']:.3f} mm",
        "",
        "## 数据与离群检查",
        "",
        "- 源图像组 ID：" + ", ".join(map(str, report["source_pair_ids"])),
    ]
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


def _next_result_directory(results_dir: Path) -> Path:
    timestamp = datetime.now()
    while True:
        candidate = results_dir / timestamp.strftime("%Y%m%d-%H%M%S%f")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        timestamp += timedelta(microseconds=1)
