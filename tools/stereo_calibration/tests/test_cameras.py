"""Hardware-free tests for synchronized stereo acquisition."""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.stereo_calibration import cameras
from tools.stereo_calibration.cameras import (
    CameraError,
    CameraWorker,
    FramePair,
    OpenCVCameraPair,
    apply_v4l2_controls,
)
from tools.stereo_calibration.config import AppConfig
from tools.stereo_calibration.detection import DetectionResult


LEFT_DEVICE = "/dev/v4l/by-path/pci-left-video-index0"
RIGHT_DEVICE = "/dev/v4l/by-path/pci-right-video-index0"


def make_config(*, controls: dict[str, int] | None = None) -> AppConfig:
    return AppConfig.from_mapping(
        {
            "left": {"name": "left", "device": LEFT_DEVICE, "rotation_degrees": 0},
            "right": {"name": "right", "device": RIGHT_DEVICE, "rotation_degrees": 180},
            "capture": {
                "width": 4,
                "height": 3,
                "fps": 30,
                "fourcc": "MJPG",
                "v4l2_controls": controls
                or {
                    "gain": 12,
                    "white_balance_automatic": 0,
                    "auto_exposure": 1,
                    "exposure_time_absolute": 80,
                },
            },
            "checkerboard": {"columns": 3, "rows": 2, "square_size_mm": 35.0},
            "quality": {
                "edge_margin_px": 0,
                "min_laplacian_variance": 0.0,
                "max_saturated_fraction": 0.9,
            },
            "data_root": "/tmp/stereo-test",
            "web": {"host": "127.0.0.1", "port": 8765},
            "minimum_pairs": 3,
            "target_pairs": 4,
            "baseline_reference_mm": 200.0,
        }
    )


class FakeCapture:
    def __init__(
        self,
        device: str,
        log: list[tuple],
        *,
        opened: bool = True,
        frame: np.ndarray | None = None,
        set_fail_property: int | None = None,
        readbacks: dict[int, float] | None = None,
        grab_results: list[bool] | None = None,
        retrieve_results: list[tuple[bool, np.ndarray | None]] | None = None,
        release_error: Exception | None = None,
    ) -> None:
        self.device = device
        self.log = log
        self.opened = opened
        self.frame = (
            np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
            if frame is None
            else frame
        )
        self.set_fail_property = set_fail_property
        self.readbacks = dict(readbacks or {})
        self.grab_results = deque(grab_results or [True])
        self.retrieve_results = deque(retrieve_results or [(True, self.frame)])
        self.release_error = release_error
        self.release_count = 0

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API spelling
        self.log.append((self.device, "isOpened"))
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.log.append((self.device, "set", prop, value))
        return prop != self.set_fail_property

    def get(self, prop: int) -> float:
        self.log.append((self.device, "get", prop))
        defaults = {
            cv2.CAP_PROP_FRAME_WIDTH: 4.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 3.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
            cv2.CAP_PROP_FPS: 30.0,
        }
        return self.readbacks.get(prop, defaults[prop])

    def grab(self) -> bool:
        self.log.append((self.device, "grab"))
        return self.grab_results.popleft() if self.grab_results else True

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        self.log.append((self.device, "retrieve"))
        return self.retrieve_results.popleft() if self.retrieve_results else (True, self.frame)

    def release(self) -> None:
        self.release_count += 1
        self.log.append((self.device, "release"))
        if self.release_error is not None:
            raise self.release_error


def configured_pair(
    *,
    left: FakeCapture | None = None,
    right: FakeCapture | None = None,
    config: AppConfig | None = None,
) -> tuple[OpenCVCameraPair, FakeCapture, FakeCapture, list[tuple]]:
    log: list[tuple] = [] if left is None else left.log
    left = left or FakeCapture(LEFT_DEVICE, log)
    right = right or FakeCapture(RIGHT_DEVICE, log)
    captures = {LEFT_DEVICE: left, RIGHT_DEVICE: right}
    pair = OpenCVCameraPair(
        config or make_config(),
        capture_factory=lambda device: captures[device],
        control_runner=lambda device, controls: log.append(
            (device, "controls", tuple(controls.items()))
        ),
    )
    return pair, left, right, log


def test_apply_controls_is_noop_for_empty_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cameras.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )
    apply_v4l2_controls(LEFT_DEVICE, {})


def test_apply_controls_rejects_blank_device_without_starting_a_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cameras.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess must not run"),
    )
    with pytest.raises(CameraError, match="device"):
        apply_v4l2_controls("   ", {"gain": 1})


@pytest.mark.parametrize(
    ("device", "controls"),
    [
        ("", {"gain": 1}),
        ("-device", {"gain": 1}),
        (LEFT_DEVICE, {"": 1}),
        (LEFT_DEVICE, {"gain=bad": 1}),
        (LEFT_DEVICE, {"gain": True}),
        (LEFT_DEVICE, {"gain": 1.5}),
    ],
)
def test_apply_controls_rejects_unsafe_inputs(device: str, controls: dict) -> None:
    with pytest.raises(CameraError):
        apply_v4l2_controls(device, controls)


def test_apply_controls_sets_modes_first_and_verifies_exact_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv: list[str], **kwargs):
        calls.append((argv, kwargs))
        output = "" if "--set-ctrl" in argv[2] else (
            "white_balance_automatic: 0\n"
            "gain: 12\n"
            "auto_exposure: 1\n"
            "exposure_time_absolute: 80\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    monkeypatch.setattr(cameras.subprocess, "run", fake_run)
    apply_v4l2_controls(
        LEFT_DEVICE,
        {
            "gain": 12,
            "exposure_time_absolute": 80,
            "white_balance_automatic": 0,
            "auto_exposure": 1,
        },
    )

    names = (
        "auto_exposure,white_balance_automatic,exposure_time_absolute,gain"
    )
    assert calls[0][0] == [
        "v4l2-ctl",
        f"--device={LEFT_DEVICE}",
        "--set-ctrl=auto_exposure=1,white_balance_automatic=0,"
        "exposure_time_absolute=80,gain=12",
    ]
    assert calls[1][0] == [
        "v4l2-ctl",
        f"--device={LEFT_DEVICE}",
        f"--get-ctrl={names}",
    ]
    assert all(
        kwargs == {"check": True, "capture_output": True, "text": True}
        for _, kwargs in calls
    )


@pytest.mark.parametrize(
    "output",
    [
        "gain: 9\n",
        "gain: 12\ngain: 12\n",
        "gain: twelve\n",
        "unrequested: 12\n",
        "",
    ],
)
def test_apply_controls_rejects_mismatch_duplicate_or_malformed_get_output(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    results = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=output, stderr="context"),
        ]
    )
    monkeypatch.setattr(cameras.subprocess, "run", lambda *a, **k: next(results))
    with pytest.raises(CameraError, match="pci-left"):
        apply_v4l2_controls(LEFT_DEVICE, {"gain": 12})


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("missing"),
        subprocess.CalledProcessError(1, ["v4l2-ctl"], stderr="denied"),
    ],
)
def test_apply_controls_wraps_process_failures(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cameras.subprocess, "run", fail)
    with pytest.raises(CameraError, match="pci-left"):
        apply_v4l2_controls(LEFT_DEVICE, {"gain": 12})


def test_open_applies_identical_controls_and_configures_both_in_order() -> None:
    pair, _, _, log = configured_pair()
    pair.open()

    assert log[0][0:2] == (LEFT_DEVICE, "controls")
    assert log[1] == (LEFT_DEVICE, "isOpened")
    right_controls_index = next(
        index for index, entry in enumerate(log) if entry[0:2] == (RIGHT_DEVICE, "controls")
    )
    left_get_indexes = [
        index for index, entry in enumerate(log) if entry[0:2] == (LEFT_DEVICE, "get")
    ]
    assert right_controls_index > max(left_get_indexes)
    assert log[right_controls_index + 1] == (RIGHT_DEVICE, "isOpened")
    assert log[0][2] == log[right_controls_index][2]
    expected_properties = [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
    ]
    assert [entry[2] for entry in log if entry[0:2] == (LEFT_DEVICE, "set")] == expected_properties
    assert [entry[2] for entry in log if entry[0:2] == (RIGHT_DEVICE, "set")] == expected_properties


def test_production_factory_uses_v4l2_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    log: list[tuple] = []
    captures = {
        LEFT_DEVICE: FakeCapture(LEFT_DEVICE, log),
        RIGHT_DEVICE: FakeCapture(RIGHT_DEVICE, log),
    }
    calls: list[tuple[str, int]] = []

    def video_capture(device: str, backend: int):
        calls.append((device, backend))
        return captures[device]

    monkeypatch.setattr(cameras.cv2, "VideoCapture", video_capture)
    pair = OpenCVCameraPair(make_config(), control_runner=lambda *args: None)
    pair.open()
    assert calls == [(LEFT_DEVICE, cv2.CAP_V4L2), (RIGHT_DEVICE, cv2.CAP_V4L2)]
    pair.close()


@pytest.mark.parametrize(
    ("readbacks", "message"),
    [
        ({cv2.CAP_PROP_FRAME_WIDTH: 5.0}, "resolution"),
        ({cv2.CAP_PROP_FRAME_HEIGHT: float("nan")}, "resolution"),
        ({cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV"))}, "FOURCC"),
        ({cv2.CAP_PROP_FOURCC: float("inf")}, "FOURCC"),
        ({cv2.CAP_PROP_FOURCC: 0.0}, "invalid FOURCC"),
        ({cv2.CAP_PROP_FPS: 27.0}, "FPS"),
        ({cv2.CAP_PROP_FPS: 0.0}, "FPS"),
    ],
)
def test_open_rejects_bad_readback_and_releases_partial_capture(
    readbacks: dict[int, float], message: str
) -> None:
    log: list[tuple] = []
    left = FakeCapture(LEFT_DEVICE, log, readbacks=readbacks)
    right = FakeCapture(RIGHT_DEVICE, log)
    pair, _, _, _ = configured_pair(left=left, right=right)
    with pytest.raises(CameraError, match=message):
        pair.open()
    assert left.release_count == 1
    assert right.release_count == 0
    with pytest.raises(CameraError, match="not open"):
        pair.read()


def test_open_failure_on_right_releases_both_and_allows_later_open() -> None:
    log: list[tuple] = []
    left = FakeCapture(LEFT_DEVICE, log)
    bad_right = FakeCapture(RIGHT_DEVICE, log, opened=False)
    captures = deque([left, bad_right])
    pair = OpenCVCameraPair(
        make_config(),
        capture_factory=lambda device: captures.popleft(),
        control_runner=lambda *args: None,
    )
    with pytest.raises(CameraError, match="right"):
        pair.open()
    assert left.release_count == bad_right.release_count == 1

    good_left = FakeCapture(LEFT_DEVICE, log)
    good_right = FakeCapture(RIGHT_DEVICE, log)
    captures.extend([good_left, good_right])
    pair.open()
    pair.close()
    assert good_left.release_count == good_right.release_count == 1


def test_open_rejects_set_failure_and_double_open_without_leak() -> None:
    log: list[tuple] = []
    failed = FakeCapture(
        LEFT_DEVICE, log, set_fail_property=cv2.CAP_PROP_FRAME_WIDTH
    )
    pair, _, right, _ = configured_pair(left=failed, right=FakeCapture(RIGHT_DEVICE, log))
    with pytest.raises(CameraError, match="set"):
        pair.open()
    assert failed.release_count == 1
    assert right.release_count == 0

    pair, left, right, _ = configured_pair()
    pair.open()
    with pytest.raises(CameraError, match="already open"):
        pair.open()
    assert left.release_count == right.release_count == 0
    pair.close()


def test_read_grabs_before_retrieve_normalizes_and_owns_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[tuple] = []
    left_raw = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    right_raw = (100 + np.arange(36, dtype=np.uint8)).reshape(3, 4, 3)
    left = FakeCapture(LEFT_DEVICE, log, frame=left_raw)
    right = FakeCapture(RIGHT_DEVICE, log, frame=right_raw)
    pair, _, _, _ = configured_pair(left=left, right=right)
    ticks = iter([100, 120])
    monkeypatch.setattr(cameras, "monotonic_ns", lambda: next(ticks))
    pair.open()
    log.clear()

    result = pair.read()

    assert log[:4] == [
        (LEFT_DEVICE, "grab"),
        (RIGHT_DEVICE, "grab"),
        (LEFT_DEVICE, "retrieve"),
        (RIGHT_DEVICE, "retrieve"),
    ]
    assert (result.left_timestamp_ns, result.right_timestamp_ns) == (100, 120)
    np.testing.assert_array_equal(result.left, left_raw)
    np.testing.assert_array_equal(result.right, cv2.rotate(right_raw, cv2.ROTATE_180))
    assert result.left.flags.owndata and result.right.flags.owndata
    left_raw[:] = 0
    right_raw[:] = 0
    assert result.left.any() and result.right.any()
    pair.close()


def test_read_normalizes_each_frame_immediately_after_its_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair, _, _, log = configured_pair()

    def tracked_normalize(frame: np.ndarray, rotation: int) -> np.ndarray:
        log.append(("normalize", rotation))
        return frame

    monkeypatch.setattr(cameras, "normalize_frame", tracked_normalize)
    pair.open()
    log.clear()
    pair.read()
    assert log == [
        (LEFT_DEVICE, "grab"),
        (RIGHT_DEVICE, "grab"),
        (LEFT_DEVICE, "retrieve"),
        ("normalize", 0),
        (RIGHT_DEVICE, "retrieve"),
        ("normalize", 180),
    ]
    pair.close()


@pytest.mark.parametrize(
    ("side", "kind"),
    [
        ("left", "grab"),
        ("right", "grab"),
        ("left", "retrieve"),
        ("right", "retrieve"),
    ],
)
def test_read_wraps_grab_and_retrieve_failures(side: str, kind: str) -> None:
    log: list[tuple] = []
    kwargs = {"grab_results": [False]} if kind == "grab" else {"retrieve_results": [(False, None)]}
    left = FakeCapture(LEFT_DEVICE, log, **(kwargs if side == "left" else {}))
    right = FakeCapture(RIGHT_DEVICE, log, **(kwargs if side == "right" else {}))
    pair, _, _, _ = configured_pair(left=left, right=right)
    pair.open()
    with pytest.raises(CameraError, match=f"{side}.*{kind}"):
        pair.read()
    pair.close()


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
    ],
)
def test_read_rejects_invalid_frames(frame: np.ndarray | None) -> None:
    log: list[tuple] = []
    left = FakeCapture(
        LEFT_DEVICE, log, retrieve_results=[(True, frame)]
    )
    pair, _, _, _ = configured_pair(left=left, right=FakeCapture(RIGHT_DEVICE, log))
    pair.open()
    with pytest.raises(CameraError, match="left.*frame"):
        pair.read()
    pair.close()


def test_read_requires_open_and_wraps_cv2_error() -> None:
    pair, left, _, _ = configured_pair()
    with pytest.raises(CameraError, match="not open"):
        pair.read()
    pair.open()
    left.grab = lambda: (_ for _ in ()).throw(cv2.error("boom"))
    with pytest.raises(CameraError, match="left.*grab"):
        pair.read()
    pair.close()


def test_close_is_idempotent_clears_state_and_reports_both_release_errors() -> None:
    log: list[tuple] = []
    left = FakeCapture(LEFT_DEVICE, log, release_error=RuntimeError("left cleanup"))
    right = FakeCapture(RIGHT_DEVICE, log, release_error=RuntimeError("right cleanup"))
    pair, _, _, _ = configured_pair(left=left, right=right)
    pair.open()
    with pytest.raises(CameraError, match="left cleanup.*right cleanup"):
        pair.close()
    assert left.release_count == right.release_count == 1
    pair.close()
    with pytest.raises(CameraError, match="not open"):
        pair.read()


def make_pair(value: int = 1) -> FramePair:
    return FramePair(
        left=np.full((3, 4, 3), value, dtype=np.uint8),
        right=np.full((3, 4, 3), value + 1, dtype=np.uint8),
        left_timestamp_ns=value * 10,
        right_timestamp_ns=value * 10 + 1,
    )


class ScriptedSource:
    def __init__(
        self,
        actions: list[FramePair | Exception],
        *,
        opened: threading.Event | None = None,
        before_read: threading.Event | None = None,
        allow_read: threading.Event | None = None,
        open_error: Exception | None = None,
        close_error: Exception | None = None,
        block_when_exhausted: threading.Event | None = None,
        exhausted: threading.Event | None = None,
    ) -> None:
        self.actions = deque(actions)
        self.opened = opened
        self.before_read = before_read
        self.allow_read = allow_read
        self.open_error = open_error
        self.close_error = close_error
        self.block_when_exhausted = block_when_exhausted
        self.exhausted = exhausted
        self.open_count = 0
        self.close_count = 0

    def open(self) -> None:
        self.open_count += 1
        if self.opened:
            self.opened.set()
        if self.open_error:
            raise self.open_error

    def read(self) -> FramePair:
        if self.before_read:
            self.before_read.set()
        if self.allow_read:
            assert self.allow_read.wait(timeout=1.0)
        if not self.actions:
            if self.exhausted:
                self.exhausted.set()
            if self.block_when_exhausted:
                assert self.block_when_exhausted.wait(timeout=1.0)
            raise CameraError("script exhausted")
        action = self.actions.popleft()
        if isinstance(action, Exception):
            raise action
        return action

    def close(self) -> None:
        self.close_count += 1
        if self.close_error:
            raise self.close_error


def successful_detection(frame: np.ndarray, board, quality) -> DetectionResult:
    corners = np.arange(board.corner_count * 2, dtype=np.float32).reshape(-1, 1, 2)
    return DetectionResult(True, corners, 10.0, 0.0, True, True, ())


def wait_for(worker: CameraWorker, predicate, *, timeout: float = 1.0):
    snapshot = worker._wait_for_snapshot(predicate, timeout)  # test-only condition wait
    assert predicate(snapshot), snapshot
    return snapshot


def test_worker_publishes_complete_pairs_and_defensive_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_exhausted = threading.Event()
    exhausted = threading.Event()
    source = ScriptedSource(
        [make_pair(1), make_pair(2)],
        block_when_exhausted=release_exhausted,
        exhausted=exhausted,
    )
    monkeypatch.setattr(cameras, "detect_checkerboard", successful_detection)
    config = make_config()
    worker = CameraWorker(lambda: source, config.checkerboard, config.quality)
    worker.start()
    snapshot = wait_for(worker, lambda item: item.sequence >= 2 and item.pair is not None)
    assert snapshot.left_detection is not None
    assert snapshot.right_detection is not None
    snapshot.pair.left[:] = 255
    snapshot.left_detection.corners[:] = 255
    fresh = worker.snapshot()
    assert not np.all(fresh.pair.left == 255)
    assert not np.all(fresh.left_detection.corners == 255)
    assert exhausted.wait(timeout=1.0)
    release_exhausted.set()
    wait_for(worker, lambda item: item.error is not None)
    worker.stop()
    assert source.close_count == 1


def test_worker_failure_clears_pair_waits_for_retry_and_recovers_fresh_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ScriptedSource([make_pair(1), CameraError("camera disconnected")])
    second_gate = threading.Event()
    second_exhausted = threading.Event()
    second = ScriptedSource(
        [make_pair(3)],
        block_when_exhausted=second_gate,
        exhausted=second_exhausted,
    )
    sources = deque([first, second])
    monkeypatch.setattr(cameras, "detect_checkerboard", successful_detection)
    config = make_config()
    worker = CameraWorker(lambda: sources.popleft(), config.checkerboard, config.quality)
    worker.start()
    error = wait_for(worker, lambda item: item.error is not None)
    assert "camera disconnected" in error.error
    assert error.pair is error.left_detection is error.right_detection is None
    assert first.close_count == 1
    assert second.open_count == 0
    worker.retry()
    recovered = wait_for(
        worker,
        lambda item: item.pair is not None and item.pair.left[0, 0, 0] == 3,
    )
    assert recovered.error is None
    assert second_exhausted.wait(timeout=1.0)
    second_gate.set()
    worker.stop()
    assert second.close_count == 1


def test_worker_retry_during_healthy_stream_closes_and_reopens_fresh_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_read = threading.Event()
    first_allow = threading.Event()
    first = ScriptedSource([make_pair(1)], before_read=first_read, allow_read=first_allow)
    second_release = threading.Event()
    second_exhausted = threading.Event()
    second = ScriptedSource(
        [make_pair(2)],
        block_when_exhausted=second_release,
        exhausted=second_exhausted,
    )
    sources = deque([first, second])
    monkeypatch.setattr(cameras, "detect_checkerboard", successful_detection)
    config = make_config()
    worker = CameraWorker(lambda: sources.popleft(), config.checkerboard, config.quality)
    worker.start()
    assert first_read.wait(timeout=1.0)
    worker.retry()
    first_allow.set()
    wait_for(worker, lambda item: item.pair is not None and item.pair.left[0, 0, 0] == 2)
    assert second_exhausted.wait(timeout=1.0)
    second_release.set()
    worker.stop()
    assert first.close_count == second.close_count == 1


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: (_ for _ in ()).throw(RuntimeError("factory boom")), "factory boom"),
        (lambda: ScriptedSource([], open_error=RuntimeError("open boom")), "open boom"),
    ],
)
def test_worker_factory_and_open_failures_are_stable_and_do_not_busy_loop(
    factory, expected
) -> None:
    config = make_config()
    calls = 0

    def counted_factory():
        nonlocal calls
        calls += 1
        return factory()

    worker = CameraWorker(counted_factory, config.checkerboard, config.quality)
    worker.start()
    snapshot = wait_for(worker, lambda item: item.error is not None)
    assert expected in snapshot.error
    assert calls == 1
    worker.stop()


def test_worker_detection_failure_preserves_primary_and_close_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ScriptedSource([make_pair()], close_error=RuntimeError("close boom"))

    def detector(*args):
        raise RuntimeError("detect boom")

    monkeypatch.setattr(cameras, "detect_checkerboard", detector)
    config = make_config()
    worker = CameraWorker(lambda: source, config.checkerboard, config.quality)
    worker.start()
    snapshot = wait_for(worker, lambda item: item.error is not None)
    assert "detect boom" in snapshot.error and "close boom" in snapshot.error
    assert snapshot.pair is None
    worker.stop()
    assert source.close_count == 1


def test_worker_stop_while_waiting_is_idempotent_and_cannot_restart() -> None:
    source = ScriptedSource([], open_error=CameraError("offline"))
    config = make_config()
    worker = CameraWorker(lambda: source, config.checkerboard, config.quality)
    worker.start()
    wait_for(worker, lambda item: item.error is not None)
    worker.stop()
    worker.stop()
    assert source.close_count == 1
    with pytest.raises(CameraError, match="start"):
        worker.start()
    with pytest.raises(CameraError, match="stopped"):
        worker.retry()


def test_worker_stop_during_read_finishes_within_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reading = threading.Event()
    allow = threading.Event()
    source = ScriptedSource([make_pair()], before_read=reading, allow_read=allow)
    monkeypatch.setattr(cameras, "detect_checkerboard", successful_detection)
    config = make_config()
    worker = CameraWorker(lambda: source, config.checkerboard, config.quality)
    worker.start()
    assert reading.wait(timeout=1.0)
    failures: list[Exception] = []

    def stop_worker() -> None:
        try:
            worker.stop()
        except Exception as exc:
            failures.append(exc)

    stopper = threading.Thread(target=stop_worker)
    stopper.start()
    assert worker._stop_event.wait(timeout=1.0)
    allow.set()
    stopper.join(timeout=1.0)
    assert not stopper.is_alive()
    assert failures == []
    assert source.close_count == 1


def test_worker_rejects_double_start_and_reports_unstoppable_thread() -> None:
    reading = threading.Event()
    allow = threading.Event()
    source = ScriptedSource([make_pair()], before_read=reading, allow_read=allow)
    config = make_config()
    worker = CameraWorker(lambda: source, config.checkerboard, config.quality)
    worker.start()
    assert reading.wait(timeout=1.0)
    with pytest.raises(CameraError, match="start"):
        worker.start()
    original_join = worker._thread.join
    worker._thread.join = lambda timeout=None: None
    worker._thread.is_alive = lambda: True
    with pytest.raises(CameraError, match="2"):
        worker.stop()
    worker._thread.join = original_join
    allow.set()
    worker._thread.join(timeout=1.0)
