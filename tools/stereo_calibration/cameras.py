"""Synchronized, hardware-isolated stereo camera acquisition.

The production camera pair owns both V4L2 handles. Browser and web-server
threads interact with CameraWorker snapshots instead of touching camera
handles directly.
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from math import isfinite
from time import monotonic, monotonic_ns
from typing import Callable, Mapping, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import AppConfig, CheckerboardConfig, QualityConfig
from .detection import DetectionResult, detect_checkerboard, normalize_frame


class CameraError(RuntimeError):
    """Raised when stereo capture cannot produce a trustworthy frame pair."""


@dataclass(frozen=True)
class FramePair:
    left: NDArray[np.uint8]
    right: NDArray[np.uint8]
    left_timestamp_ns: int
    right_timestamp_ns: int


class PairSource(Protocol):
    """Minimal source interface exclusively owned by CameraWorker."""

    def open(self) -> None: ...

    def read(self) -> FramePair: ...

    def close(self) -> None: ...


_CONTROL_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_CONTROL_OUTPUT = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+)\s*$")
_MODE_CONTROLS = ("auto_exposure", "white_balance_automatic")


def _validate_control_inputs(device: str, controls: Mapping[str, int]) -> list[str]:
    if (
        not isinstance(device, str)
        or not device.strip()
        or not device.startswith("/dev/")
        or any(ord(character) < 32 for character in device)
    ):
        raise CameraError(f"invalid V4L2 device {device!r}")
    if not isinstance(controls, Mapping):
        raise CameraError(f"invalid V4L2 controls for {device}: expected a mapping")

    for name, value in controls.items():
        if not isinstance(name, str) or _CONTROL_NAME.fullmatch(name) is None:
            raise CameraError(f"invalid V4L2 control name {name!r} for {device}")
        if type(value) is not int:
            raise CameraError(f"invalid V4L2 control value for {name!r} on {device}")

    modes = [name for name in _MODE_CONTROLS if name in controls]
    remaining = sorted(name for name in controls if name not in _MODE_CONTROLS)
    return modes + remaining


def apply_v4l2_controls(device: str, controls: Mapping[str, int]) -> None:
    """Apply and read back V4L2 controls using shell-free subprocess calls.

    An empty mapping is deliberately a no-op, which is useful for cameras that
    expose no portable controls. Non-empty mappings are set in deterministic
    order with mode switches preceding their dependent manual values.
    """

    names = _validate_control_inputs(device, controls)
    if not names:
        return

    assignments = ",".join(f"{name}={controls[name]}" for name in names)
    requested_names = ",".join(names)
    commands = (
        ["v4l2-ctl", f"--device={device}", f"--set-ctrl={assignments}"],
        ["v4l2-ctl", f"--device={device}", f"--get-ctrl={requested_names}"],
    )
    results: list[subprocess.CompletedProcess[str]] = []
    try:
        for command in commands:
            results.append(
                subprocess.run(command, check=True, capture_output=True, text=True)
            )
    except FileNotFoundError as exc:
        raise CameraError(f"could not configure {device}: v4l2-ctl was not found") from exc
    except subprocess.CalledProcessError as exc:
        context = (exc.stderr or exc.stdout or str(exc)).strip()
        raise CameraError(f"could not configure {device}: {context}") from exc
    except (OSError, UnicodeError) as exc:
        raise CameraError(f"could not configure {device}: {exc}") from exc

    output = results[-1].stdout
    returned: dict[str, int] = {}
    try:
        lines = output.splitlines()
    except (AttributeError, UnicodeError) as exc:
        raise CameraError(f"malformed V4L2 control output for {device}: {output!r}") from exc
    for line in lines:
        match = _CONTROL_OUTPUT.fullmatch(line)
        if match is None:
            raise CameraError(f"malformed V4L2 control output for {device}: {line!r}")
        name, raw_value = match.groups()
        if name not in controls:
            raise CameraError(f"unexpected V4L2 control {name!r} returned for {device}")
        if name in returned:
            raise CameraError(f"duplicate V4L2 control {name!r} returned for {device}")
        returned[name] = int(raw_value)

    for name in names:
        if name not in returned:
            context = (results[-1].stderr or output or "empty output").strip()
            raise CameraError(
                f"V4L2 control {name!r} missing from {device} readback: {context}"
            )
        if returned[name] != controls[name]:
            raise CameraError(
                f"V4L2 control {name!r} mismatch on {device}: "
                f"expected {controls[name]}, got {returned[name]}"
            )


def _release_capture(capture: object, label: str) -> str | None:
    try:
        capture.release()  # type: ignore[attr-defined]
    except Exception as exc:
        return f"{label} release failed: {exc}"
    return None


def _fourcc_text(value: float, device: str) -> str:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise CameraError(f"invalid FOURCC readback from {device}: {value!r}")
    integer = int(round(value))
    if abs(float(value) - integer) > 1e-6:
        raise CameraError(f"invalid FOURCC readback from {device}: {value!r}")
    decoded = "".join(chr((integer >> (8 * offset)) & 0xFF) for offset in range(4))
    if any(not 32 <= ord(character) <= 126 for character in decoded):
        raise CameraError(f"invalid FOURCC readback from {device}: {value!r}")
    return decoded


class OpenCVCameraPair:
    """Own and synchronously acquire a configured pair of OpenCV V4L2 cameras."""

    def __init__(
        self,
        config: AppConfig,
        capture_factory: Callable[[str], object] | None = None,
        control_runner: Callable[[str, Mapping[str, int]], None] = apply_v4l2_controls,
    ) -> None:
        self._config = config
        self._capture_factory = capture_factory
        self._control_runner = control_runner
        self._left: object | None = None
        self._right: object | None = None

    def _make_capture(self, device: str) -> object:
        if self._capture_factory is not None:
            return self._capture_factory(device)
        return cv2.VideoCapture(device, cv2.CAP_V4L2)

    def _configure_capture(self, capture: object, device: str, label: str) -> None:
        try:
            if not capture.isOpened():  # type: ignore[attr-defined]
                raise CameraError(f"{label} camera did not open: {device}")

            settings = (
                (cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*self._config.capture.fourcc))),
                (cv2.CAP_PROP_FRAME_WIDTH, float(self._config.capture.width)),
                (cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.capture.height)),
                (cv2.CAP_PROP_FPS, float(self._config.capture.fps)),
                (cv2.CAP_PROP_BUFFERSIZE, 1.0),
            )
            for prop, value in settings:
                result = capture.set(prop, value)  # type: ignore[attr-defined]
                if isinstance(result, (bool, np.bool_)) and not bool(result):
                    raise CameraError(
                        f"{label} camera failed to set property {prop} on {device}"
                    )

            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # type: ignore[attr-defined]
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type: ignore[attr-defined]
            if (
                type(width) not in (int, float)
                or type(height) not in (int, float)
                or not isfinite(width)
                or not isfinite(height)
                or width <= 0
                or height <= 0
                or abs(width - self._config.capture.width) > 0.5
                or abs(height - self._config.capture.height) > 0.5
            ):
                raise CameraError(
                    f"{label} camera resolution mismatch on {device}: "
                    f"expected {self._config.capture.image_size}, got {(width, height)}"
                )

            raw_fourcc = capture.get(cv2.CAP_PROP_FOURCC)  # type: ignore[attr-defined]
            actual_fourcc = _fourcc_text(raw_fourcc, device)
            if actual_fourcc != self._config.capture.fourcc:
                raise CameraError(
                    f"{label} camera FOURCC mismatch on {device}: "
                    f"expected {self._config.capture.fourcc}, got {actual_fourcc!r}"
                )

            fps = capture.get(cv2.CAP_PROP_FPS)  # type: ignore[attr-defined]
            if (
                type(fps) not in (int, float)
                or not isfinite(fps)
                or fps <= 0
                or abs(fps - self._config.capture.fps) > 1.0
            ):
                raise CameraError(
                    f"{label} camera FPS mismatch on {device}: "
                    f"expected {self._config.capture.fps}, got {fps!r}"
                )
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(f"could not configure {label} camera {device}: {exc}") from exc

    def open(self) -> None:
        if self._left is not None or self._right is not None:
            raise CameraError("stereo camera pair is already open")

        opened: list[tuple[object, str]] = []
        try:
            self._control_runner(
                self._config.left.device, self._config.capture.v4l2_controls
            )
            left = self._make_capture(self._config.left.device)
            opened.append((left, "left"))
            self._configure_capture(left, self._config.left.device, "left")

            self._control_runner(
                self._config.right.device, self._config.capture.v4l2_controls
            )
            right = self._make_capture(self._config.right.device)
            opened.append((right, "right"))
            self._configure_capture(right, self._config.right.device, "right")
        except Exception as exc:
            cleanup = [
                error
                for capture, label in reversed(opened)
                if (error := _release_capture(capture, label)) is not None
            ]
            self._left = self._right = None
            primary = exc if isinstance(exc, CameraError) else CameraError(str(exc))
            message = str(primary)
            if cleanup:
                message = f"{message}; cleanup errors: {'; '.join(cleanup)}"
            raise CameraError(message) from exc

        self._left = left
        self._right = right

    def close(self) -> None:
        left, right = self._left, self._right
        self._left = self._right = None
        errors: list[str] = []
        if left is not None:
            error = _release_capture(left, "left")
            if error:
                errors.append(error)
        if right is not None:
            error = _release_capture(right, "right")
            if error:
                errors.append(error)
        if errors:
            raise CameraError("; ".join(errors))

    def _retrieve(self, capture: object, label: str) -> tuple[NDArray[np.uint8], int]:
        try:
            ok, frame = capture.retrieve()  # type: ignore[attr-defined]
        except Exception as exc:
            raise CameraError(f"{label} camera retrieve failed: {exc}") from exc
        timestamp = monotonic_ns()
        if not ok:
            raise CameraError(f"{label} camera retrieve failed")
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[:2]
            != (self._config.capture.height, self._config.capture.width)
        ):
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            raise CameraError(
                f"{label} camera returned invalid frame: shape={shape}, dtype={dtype}"
            )
        return frame, timestamp

    def _normalize(
        self,
        frame: NDArray[np.uint8],
        rotation_degrees: int,
        label: str,
    ) -> NDArray[np.uint8]:
        try:
            normalized = normalize_frame(frame, rotation_degrees)
        except Exception as exc:
            raise CameraError(f"could not normalize {label} frame: {exc}") from exc
        expected_shape = (
            self._config.capture.height,
            self._config.capture.width,
            3,
        )
        if normalized.shape != expected_shape:
            raise CameraError(
                f"normalized {label} frame dimensions do not match {expected_shape}: "
                f"got {normalized.shape}"
            )
        return np.array(normalized, dtype=np.uint8, copy=True)

    def read(self) -> FramePair:
        if self._left is None or self._right is None:
            raise CameraError("stereo camera pair is not open")
        try:
            left_ok = self._left.grab()  # type: ignore[attr-defined]
        except Exception as exc:
            raise CameraError(f"left camera grab failed: {exc}") from exc
        if not left_ok:
            raise CameraError("left camera grab failed")
        try:
            right_ok = self._right.grab()  # type: ignore[attr-defined]
        except Exception as exc:
            raise CameraError(f"right camera grab failed: {exc}") from exc
        if not right_ok:
            raise CameraError("right camera grab failed")

        left_raw, left_timestamp = self._retrieve(self._left, "left")
        left = self._normalize(left_raw, self._config.left.rotation_degrees, "left")
        right_raw, right_timestamp = self._retrieve(self._right, "right")
        right = self._normalize(right_raw, self._config.right.rotation_degrees, "right")
        if right_timestamp < left_timestamp:
            raise CameraError("camera retrieval timestamps are not monotonic")
        return FramePair(
            left=left,
            right=right,
            left_timestamp_ns=left_timestamp,
            right_timestamp_ns=right_timestamp,
        )


@dataclass(frozen=True)
class WorkerSnapshot:
    sequence: int
    pair: FramePair | None
    left_detection: DetectionResult | None
    right_detection: DetectionResult | None
    error: str | None


def _copy_detection(result: DetectionResult | None) -> DetectionResult | None:
    if result is None:
        return None
    corners = None if result.corners is None else np.array(result.corners, copy=True)
    return DetectionResult(
        found=result.found,
        corners=corners,
        sharpness=result.sharpness,
        saturated_fraction=result.saturated_fraction,
        border_ok=result.border_ok,
        saveable=result.saveable,
        reasons=tuple(result.reasons),
    )


def _copy_pair(pair: FramePair | None) -> FramePair | None:
    if pair is None:
        return None
    return FramePair(
        left=np.array(pair.left, dtype=np.uint8, copy=True),
        right=np.array(pair.right, dtype=np.uint8, copy=True),
        left_timestamp_ns=pair.left_timestamp_ns,
        right_timestamp_ns=pair.right_timestamp_ns,
    )


class CameraWorker:
    """Continuously acquire and detect frames in one camera-owning daemon thread."""

    def __init__(
        self,
        source_factory: Callable[[], PairSource],
        board: CheckerboardConfig,
        quality: QualityConfig,
    ) -> None:
        self._source_factory = source_factory
        self._board = board
        self._quality = quality
        self._condition = threading.Condition()
        self._retry_event = threading.Event()
        self._stop_event = threading.Event()
        self._snapshot = WorkerSnapshot(0, None, None, None, None)
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False

    def start(self) -> None:
        with self._condition:
            if self._started:
                raise CameraError("CameraWorker can only start once")
            self._started = True
            self._stopped = False
            self._stop_event.clear()
            self._retry_event.set()
            self._thread = threading.Thread(
                target=self._run,
                name="stereo-camera-worker",
                daemon=True,
            )
            self._thread.start()

    def _publish(
        self,
        pair: FramePair | None,
        left_detection: DetectionResult | None,
        right_detection: DetectionResult | None,
        error: str | None,
    ) -> None:
        with self._condition:
            self._snapshot = WorkerSnapshot(
                self._snapshot.sequence + 1,
                pair,
                left_detection,
                right_detection,
                error,
            )
            self._condition.notify_all()

    @staticmethod
    def _error_text(primary: Exception, cleanup: Exception | None = None) -> str:
        message = str(primary) or primary.__class__.__name__
        if cleanup is not None:
            message += f"; source close failed: {cleanup}"
        return message

    def _run(self) -> None:
        while True:
            self._retry_event.wait()
            if self._stop_event.is_set():
                return
            self._retry_event.clear()

            source: PairSource | None = None
            primary_error: Exception | None = None
            close_error: Exception | None = None
            try:
                source = self._source_factory()
                source.open()
                while not self._stop_event.is_set() and not self._retry_event.is_set():
                    pair = source.read()
                    if self._stop_event.is_set() or self._retry_event.is_set():
                        break
                    left_detection = detect_checkerboard(
                        pair.left, self._board, self._quality
                    )
                    right_detection = detect_checkerboard(
                        pair.right, self._board, self._quality
                    )
                    if self._stop_event.is_set() or self._retry_event.is_set():
                        break
                    self._publish(pair, left_detection, right_detection, None)
            except Exception as exc:
                primary_error = exc
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception as exc:
                        close_error = exc

            if primary_error is not None:
                self._publish(
                    None,
                    None,
                    None,
                    self._error_text(primary_error, close_error),
                )
            elif close_error is not None:
                self._publish(None, None, None, f"source close failed: {close_error}")

            if self._stop_event.is_set():
                return
            # A healthy retry leaves the event set and immediately constructs a
            # fresh source. An error leaves it clear and waits for user action.

    def snapshot(self) -> WorkerSnapshot:
        with self._condition:
            snapshot = self._snapshot
            return WorkerSnapshot(
                snapshot.sequence,
                _copy_pair(snapshot.pair),
                _copy_detection(snapshot.left_detection),
                _copy_detection(snapshot.right_detection),
                snapshot.error,
            )

    def _wait_for_snapshot(
        self,
        predicate: Callable[[WorkerSnapshot], bool],
        timeout: float,
    ) -> WorkerSnapshot:
        """Condition-based wait used by deterministic tests and internal callers."""

        deadline = monotonic() + timeout
        with self._condition:
            while not predicate(self._snapshot):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            snapshot = self._snapshot
            return WorkerSnapshot(
                snapshot.sequence,
                _copy_pair(snapshot.pair),
                _copy_detection(snapshot.left_detection),
                _copy_detection(snapshot.right_detection),
                snapshot.error,
            )

    def retry(self) -> None:
        with self._condition:
            if not self._started:
                raise CameraError("CameraWorker has not started")
            if self._stopped or self._stop_event.is_set():
                raise CameraError("CameraWorker is stopped")
            self._retry_event.set()

    def stop(self) -> None:
        with self._condition:
            if self._stopped:
                return
            if not self._started:
                self._stopped = True
                return
            thread = self._thread
            self._stop_event.set()
            self._retry_event.set()
        assert thread is not None
        thread.join(timeout=2.0)
        if thread.is_alive():
            raise CameraError("CameraWorker did not stop within 2 seconds")
        with self._condition:
            self._stopped = True
