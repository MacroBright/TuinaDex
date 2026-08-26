"""Thread-safe application layer for guided stereo calibration."""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .calibration import (
    calibrate_observations,
    load_session_observations,
    write_calibration_run,
)
from .cameras import FramePair, WorkerSnapshot
from .config import AppConfig, CameraConfig
from .detection import DetectionResult
from .session import SavedPair, SessionStore


class ServiceError(RuntimeError):
    """Raised when a web action cannot be completed safely."""


_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s:;,]+/?)+")


class CalibrationService:
    """Coordinate snapshots, durable sessions, and background calibration jobs."""

    def __init__(self, worker: Any, session: SessionStore, config: AppConfig) -> None:
        self._worker = worker
        self._session = session
        self._config = config
        self._action_lock = threading.RLock()
        self._job_lock = threading.Lock()
        self._last_captured_sequence: int | None = None
        self._job_counter = 0
        self._calibration_state = "idle"
        self._calibration_job_id: str | None = None
        self._calibration_report_path: str | None = None
        self._calibration_error: str | None = None
        self._calibration_thread: threading.Thread | None = None

    def status(self) -> dict[str, Any]:
        """Return one coherent, JSON-safe view of cameras, session, and job state."""
        try:
            snapshot = self._worker.snapshot()
            active_count = len(self._session.active_pairs())
        except Exception as exc:
            raise ServiceError(f"could not read service status: {_safe_detail(exc)}") from exc

        with self._job_lock:
            calibration = {
                "state": self._calibration_state,
                "job_id": self._calibration_job_id,
                "report_path": self._calibration_report_path,
                "error": self._calibration_error,
            }

        complete_pair = _pair_has_expected_images(snapshot.pair, self._config)
        left_ready = _detection_is_capture_ready(
            snapshot.left_detection, self._config.checkerboard.corner_count
        )
        right_ready = _detection_is_capture_ready(
            snapshot.right_detection, self._config.checkerboard.corner_count
        )
        can_capture = bool(
            snapshot.error is None and complete_pair and left_ready and right_ready
        )
        left_status = _camera_status(
            self._config.left, snapshot.left_detection, snapshot
        )
        right_status = _camera_status(
            self._config.right, snapshot.right_detection, snapshot
        )
        return {
            "session_id": self._session.session_dir.name,
            "sequence": _plain_integer(snapshot.sequence),
            "active_count": active_count,
            "target_pairs": self._config.target_pairs,
            "can_capture": can_capture,
            "camera_error": None
            if snapshot.error is None
            else _public_text(snapshot.error),
            "left": left_status,
            "right": right_status,
            "calibration": calibration,
        }

    def preview_jpeg(self) -> bytes:
        """Render the latest normalized worker snapshot without reading cameras."""
        try:
            snapshot = self._worker.snapshot()
        except Exception as exc:
            raise ServiceError(f"could not read camera preview: {_safe_detail(exc)}") from exc
        if snapshot.pair is None:
            raise ServiceError("a complete stereo pair is not available")

        try:
            left, right = _validated_pair_images(snapshot.pair, self._config)
            left = np.array(left, copy=True)
            right = np.array(right, copy=True)
            _annotate_preview_side(
                left,
                "LEFT",
                self._config.left,
                self._config.checkerboard.pattern_size,
                snapshot.left_detection,
            )
            _annotate_preview_side(
                right,
                "RIGHT",
                self._config.right,
                self._config.checkerboard.pattern_size,
                snapshot.right_detection,
            )
            canvas = np.hstack((left, right))
            divider_x = left.shape[1]
            cv2.line(canvas, (divider_x, 0), (divider_x, canvas.shape[0] - 1), (255, 255, 255), 2)
            if snapshot.error:
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 25), (0, 0, 180), -1)
                cv2.putText(
                    canvas,
                    f"CAMERA ERROR: {_public_text(snapshot.error)[:80]}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 4), (0, 160, 0), -1)
            ok, encoded = cv2.imencode(
                ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
        except ServiceError:
            raise
        except (cv2.error, TypeError, ValueError) as exc:
            raise ServiceError(f"could not render camera preview: {_safe_detail(exc)}") from exc
        if not ok or not isinstance(encoded, np.ndarray):
            raise ServiceError("preview encoding failed")
        payload = encoded.tobytes()
        if not isinstance(payload, bytes) or not payload:
            raise ServiceError("preview encoding failed")
        return payload

    def capture(self) -> SavedPair:
        """Durably save exactly one validated worker snapshot."""
        with self._action_lock:
            try:
                snapshot = self._worker.snapshot()
            except Exception as exc:
                raise ServiceError(f"could not read camera snapshot: {_safe_detail(exc)}") from exc
            sequence = _plain_integer(snapshot.sequence)
            if self._last_captured_sequence == sequence:
                raise ServiceError(f"worker sequence {sequence} was already captured")
            try:
                pair, left_corners, right_corners = self._validated_capture(snapshot)
                metadata = {
                    "worker_sequence": sequence,
                    "timestamps_ns": {
                        "left": pair.left_timestamp_ns,
                        "right": pair.right_timestamp_ns,
                    },
                    "image_size": [self._config.capture.width, self._config.capture.height],
                    "left_corners": left_corners.reshape(-1, 2).tolist(),
                    "right_corners": right_corners.reshape(-1, 2).tolist(),
                    "quality": {
                        "left": _audit_quality(snapshot.left_detection),
                        "right": _audit_quality(snapshot.right_detection),
                    },
                }
                saved = self._session.save_pair(
                    np.array(pair.left, copy=True),
                    np.array(pair.right, copy=True),
                    metadata,
                )
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError(f"could not save camera pair: {_safe_detail(exc)}") from exc
            self._last_captured_sequence = sequence
            return saved

    def _validated_capture(
        self, snapshot: WorkerSnapshot
    ) -> tuple[FramePair, NDArray[np.float32], NDArray[np.float32]]:
        if snapshot.error is not None:
            raise ServiceError(f"camera worker is unhealthy: {_public_text(snapshot.error)}")
        if snapshot.pair is None:
            raise ServiceError("a complete stereo pair is not available")
        _validated_pair_images(snapshot.pair, self._config)
        _validate_timestamps(snapshot.pair)
        left = _validated_corners(
            snapshot.left_detection,
            self._config.checkerboard.corner_count,
            "left",
        )
        right = _validated_corners(
            snapshot.right_detection,
            self._config.checkerboard.corner_count,
            "right",
        )
        return snapshot.pair, left, right

    def reject_last(self) -> SavedPair:
        """Reject the latest active pair using the operator audit reason."""
        with self._action_lock:
            try:
                return self._session.reject_last("operator requested deletion")
            except Exception as exc:
                raise ServiceError(f"could not reject last pair: {_safe_detail(exc)}") from exc

    def retry_cameras(self) -> None:
        """Ask the worker to reconnect without touching session data."""
        with self._action_lock:
            try:
                self._worker.retry()
            except Exception as exc:
                raise ServiceError(f"could not retry cameras: {_safe_detail(exc)}") from exc

    def start_calibration(self) -> str:
        """Start one daemon calibration job and return its stable identifier."""
        # Check before waiting on the action lock so a second request conflicts
        # promptly while the running job owns that lock.
        with self._job_lock:
            if self._calibration_state == "running":
                raise ServiceError("calibration is already running")
        with self._action_lock:
            try:
                active_count = len(self._session.active_pairs())
            except Exception as exc:
                raise ServiceError(
                    f"could not inspect active pairs: {_safe_detail(exc)}"
                ) from exc
            if active_count < self._config.minimum_pairs:
                raise ServiceError(
                    f"only {active_count} active pairs; "
                    f"{self._config.minimum_pairs} required"
                )
        with self._job_lock:
            if self._calibration_state == "running":
                raise ServiceError("calibration is already running")
            self._job_counter += 1
            job_id = f"calibration-{self._job_counter:04d}"
            self._calibration_state = "running"
            self._calibration_job_id = job_id
            self._calibration_report_path = None
            self._calibration_error = None
            thread = threading.Thread(
                target=self._run_calibration,
                args=(job_id,),
                name=job_id,
                daemon=True,
            )
            self._calibration_thread = thread
            try:
                thread.start()
            except Exception as exc:
                self._calibration_state = "error"
                self._calibration_error = _safe_detail(exc)
                self._calibration_thread = None
                raise ServiceError(
                    f"could not start calibration: {_safe_detail(exc)}"
                ) from exc
        return job_id

    def _run_calibration(self, job_id: str) -> None:
        try:
            with self._action_lock:
                observations, diagnostics = load_session_observations(
                    self._session, self._config.checkerboard
                )
                if len(observations) < self._config.minimum_pairs:
                    detail = _diagnostic_text(diagnostics)
                    message = (
                        f"only {len(observations)} valid pairs; "
                        f"{self._config.minimum_pairs} required"
                    )
                    if detail:
                        message += f"; skipped: {detail}"
                    raise ServiceError(message)
                active = {pair.pair_id: pair for pair in self._session.active_pairs()}
                observation_ids = [observation.pair_id for observation in observations]
                try:
                    source_pairs = [active[pair_id] for pair_id in observation_ids]
                except KeyError as exc:
                    raise ServiceError("active source pairs changed during calibration") from exc
                if len(source_pairs) != len(observations):
                    raise ServiceError("valid calibration sources are incomplete")
                result = calibrate_observations(
                    observations,
                    self._config.capture.image_size,
                    self._config.minimum_pairs,
                )
                run_dir = write_calibration_run(
                    self._session, result, self._config, source_pairs
                )
            report_path = Path(run_dir) / "report.json"
            report = _read_report(report_path)
            state = report.get("status")
            if state not in ("pass", "warning", "fail"):
                raise ServiceError("calibration report contains an invalid status")
            try:
                display_path = report_path.relative_to(self._session.session_dir).as_posix()
            except ValueError as exc:
                raise ServiceError("calibration report was written outside the session") from exc
            with self._job_lock:
                if self._calibration_job_id == job_id:
                    self._calibration_state = state
                    self._calibration_report_path = display_path
                    self._calibration_error = None
        except Exception as exc:
            with self._job_lock:
                if self._calibration_job_id == job_id:
                    self._calibration_state = "error"
                    self._calibration_report_path = None
                    self._calibration_error = _safe_detail(exc)


def _camera_status(
    camera: CameraConfig,
    detection: DetectionResult | None,
    snapshot: WorkerSnapshot,
) -> dict[str, Any]:
    if snapshot.error is not None:
        health = "error"
    elif snapshot.pair is None:
        health = "waiting"
    else:
        health = "ready"
    return {
        "device": camera.device,
        "label": camera.name,
        "rotation_degrees": camera.rotation_degrees,
        "health": health,
        "found": bool(detection.found) if detection is not None else False,
        "saveable": bool(detection.saveable) if detection is not None else False,
        "corner_count": detection.corner_count if detection is not None else 0,
        "sharpness": _finite_or_none(detection.sharpness) if detection is not None else None,
        "saturation": (
            _finite_or_none(detection.saturated_fraction)
            if detection is not None
            else None
        ),
        "border_ok": bool(detection.border_ok) if detection is not None else False,
        "reasons": [_public_text(reason) for reason in detection.reasons]
        if detection is not None
        else [],
    }


def _audit_quality(detection: DetectionResult | None) -> dict[str, Any]:
    if detection is None:
        raise ServiceError("checkerboard detection is missing")
    return {
        "found": bool(detection.found),
        "saveable": bool(detection.saveable),
        "corner_count": detection.corner_count,
        "sharpness": _finite_or_none(detection.sharpness),
        "saturation": _finite_or_none(detection.saturated_fraction),
        "border_ok": bool(detection.border_ok),
        "reasons": [_sanitize_text(reason) for reason in detection.reasons],
    }


def _detection_is_capture_ready(
    detection: DetectionResult | None, expected_count: int
) -> bool:
    if detection is None or not detection.saveable:
        return False
    try:
        _validated_corners(detection, expected_count, "camera")
    except ServiceError:
        return False
    return True


def _validated_corners(
    detection: DetectionResult | None, expected_count: int, side: str
) -> NDArray[np.float32]:
    if detection is None:
        raise ServiceError(f"{side} checkerboard detection is missing")
    if not detection.saveable:
        reasons = ", ".join(_public_text(reason) for reason in detection.reasons)
        suffix = f": {reasons}" if reasons else ""
        raise ServiceError(f"{side} checkerboard is not saveable{suffix}")
    corners = detection.corners
    if not isinstance(corners, np.ndarray) or corners.shape != (expected_count, 1, 2):
        raise ServiceError(
            f"{side} checkerboard must contain exactly {expected_count} shaped corners"
        )
    if not np.issubdtype(corners.dtype, np.number) or not np.all(np.isfinite(corners)):
        raise ServiceError(f"{side} checkerboard corners must be finite")
    return np.asarray(corners, dtype=np.float32)


def _pair_has_expected_images(pair: FramePair | None, config: AppConfig) -> bool:
    if pair is None:
        return False
    try:
        _validated_pair_images(pair, config)
        _validate_timestamps(pair)
    except ServiceError:
        return False
    return True


def _validated_pair_images(
    pair: FramePair, config: AppConfig
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    expected = (config.capture.height, config.capture.width, 3)
    images: list[NDArray[np.uint8]] = []
    for side, image in (("left", pair.left), ("right", pair.right)):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape != expected
        ):
            raise ServiceError(
                f"{side} image must be an {expected[1]}x{expected[0]} BGR uint8 frame"
            )
        images.append(image)
    return images[0], images[1]


def _validate_timestamps(pair: FramePair) -> None:
    left = _plain_integer(pair.left_timestamp_ns)
    right = _plain_integer(pair.right_timestamp_ns)
    if left < 0 or right < left:
        raise ServiceError("camera timestamps are invalid")


def _plain_integer(value: Any) -> int:
    if type(value) is not int:
        raise ServiceError("worker sequence and timestamps must be integers")
    return value


def _annotate_preview_side(
    image: NDArray[np.uint8],
    side: str,
    camera: CameraConfig,
    pattern_size: tuple[int, int],
    detection: DetectionResult | None,
) -> None:
    if detection is not None and detection.corners is not None:
        corners = detection.corners
        expected = (pattern_size[0] * pattern_size[1], 1, 2)
        if (
            isinstance(corners, np.ndarray)
            and corners.shape == expected
            and expected[0] > 0
            and np.all(np.isfinite(corners))
        ):
            # Shape validation prevents malformed worker data from entering OpenCV.
            cv2.drawChessboardCorners(image, pattern_size, corners, detection.found)
    saveable = bool(detection is not None and detection.saveable)
    color = (40, 220, 40) if saveable else (40, 40, 230)
    cv2.putText(
        image,
        f"{side} {camera.name} ROT={camera.rotation_degrees}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )
    quality = "waiting"
    if detection is not None:
        quality = (
            f"corners={detection.corner_count} sharp={_display_number(detection.sharpness)} "
            f"sat={_display_number(detection.saturated_fraction)} "
            f"{'SAVE' if saveable else 'REJECT'}"
        )
    cv2.putText(
        image,
        quality[:70],
        (6, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        color,
        1,
        cv2.LINE_AA,
    )
    if detection is not None and detection.reasons:
        cv2.putText(
            image,
            "; ".join(_public_text(item) for item in detection.reasons)[:70],
            (6, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )


def _display_number(value: Any) -> str:
    finite = _finite_or_none(value)
    return "n/a" if finite is None else f"{finite:.3f}"


def _finite_or_none(value: Any) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _sanitize_text(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text[:500]


def _public_text(value: Any) -> str:
    return _ABSOLUTE_PATH.sub("<path>", _sanitize_text(value))


def _safe_detail(exc: BaseException) -> str:
    message = _public_text(str(exc) or exc.__class__.__name__)
    if isinstance(exc, ServiceError):
        return message
    return f"{exc.__class__.__name__}: {message}"


def _diagnostic_text(diagnostics: list[dict[str, str | int]]) -> str:
    items: list[str] = []
    for diagnostic in diagnostics[:20]:
        pair_id = diagnostic.get("pair_id", "?")
        reason = _sanitize_text(diagnostic.get("reason", "unknown reason"))
        items.append(f"pair {pair_id}: {reason}")
    if len(diagnostics) > 20:
        items.append(f"{len(diagnostics) - 20} more")
    return "; ".join(items)


def _read_report(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ServiceError("could not read the calibration report") from exc
    if not isinstance(payload, dict):
        raise ServiceError("calibration report must contain an object")
    return payload
