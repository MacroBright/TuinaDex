"""Hardware-free tests for the guided stereo calibration web application."""

from __future__ import annotations

import json
import http.client
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.stereo_calibration import service as service_module
from tools.stereo_calibration.cameras import FramePair, WorkerSnapshot
from tools.stereo_calibration.config import AppConfig
from tools.stereo_calibration.detection import DetectionResult
from tools.stereo_calibration.service import CalibrationService, ServiceError
from tools.stereo_calibration.session import SessionStore
from tools.stereo_calibration.web import build_server


LEFT_DEVICE = "/dev/v4l/by-path/pci-left-video-index0"
RIGHT_DEVICE = "/dev/v4l/by-path/pci-right-video-index0"


def make_config(
    tmp_path: Path,
    *,
    minimum_pairs: int = 3,
    left_rotation: int = 0,
    right_rotation: int = 180,
) -> AppConfig:
    return AppConfig.from_mapping(
        {
            "left": {
                "name": "cam-left",
                "device": LEFT_DEVICE,
                "rotation_degrees": left_rotation,
            },
            "right": {
                "name": "cam-right",
                "device": RIGHT_DEVICE,
                "rotation_degrees": right_rotation,
            },
            "capture": {
                "width": 160,
                "height": 120,
                "fps": 30,
                "fourcc": "MJPG",
                "v4l2_controls": {"gain": 12},
            },
            "checkerboard": {"columns": 3, "rows": 2, "square_size_mm": 35.0},
            "quality": {
                "edge_margin_px": 2,
                "min_laplacian_variance": 1.0,
                "max_saturated_fraction": 0.9,
            },
            "data_root": str(tmp_path / "data"),
            "web": {"host": "127.0.0.1", "port": 8765},
            "minimum_pairs": minimum_pairs,
            "target_pairs": max(4, minimum_pairs),
            "baseline_reference_mm": 200.0,
        }
    )


def good_corners() -> np.ndarray:
    return np.array(
        [[[25.0, 30.0]], [[60.0, 30.0]], [[95.0, 30.0]],
         [[25.0, 65.0]], [[60.0, 65.0]], [[95.0, 65.0]]],
        dtype=np.float32,
    )


def detection(
    *, saveable: bool = True, reasons: tuple[str, ...] = (), corners: np.ndarray | None = None
) -> DetectionResult:
    if corners is None:
        corners = good_corners()
    return DetectionResult(
        found=True,
        corners=corners,
        sharpness=123.25,
        saturated_fraction=0.0125,
        border_ok=True,
        saveable=saveable,
        reasons=reasons,
    )


def good_snapshot(sequence: int = 7) -> WorkerSnapshot:
    left = np.zeros((120, 160, 3), dtype=np.uint8)
    left[:, :, 1] = 40
    right = np.zeros((120, 160, 3), dtype=np.uint8)
    right[:, :, 2] = 80
    return WorkerSnapshot(
        sequence=sequence,
        pair=FramePair(left, right, 1_000_000, 1_000_100),
        left_detection=detection(),
        right_detection=detection(),
        error=None,
    )


class FakeWorker:
    def __init__(self, snapshot: WorkerSnapshot | None = None) -> None:
        self.current = snapshot or good_snapshot()
        self.snapshot_calls = 0
        self.retry_calls = 0

    def snapshot(self) -> WorkerSnapshot:
        self.snapshot_calls += 1
        return self.current

    def retry(self) -> None:
        self.retry_calls += 1


def make_service(
    tmp_path: Path,
    snapshot: WorkerSnapshot | None = None,
    *,
    config: AppConfig | None = None,
):
    config = config or make_config(tmp_path)
    store = SessionStore.create(config.data_root, "session-001")
    worker = FakeWorker(snapshot)
    return CalibrationService(worker, store, config), worker, store, config


def wait_for_state(service: CalibrationService, expected: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = service.status()
        if status["calibration"]["state"] == expected:
            return status
        time.sleep(0.01)
    raise AssertionError(f"calibration did not reach state {expected!r}")


def test_status_uses_one_snapshot_and_reports_json_safe_quality(tmp_path: Path) -> None:
    service, worker, _store, _config = make_service(tmp_path)
    worker.current = replace(
        worker.current,
        left_detection=replace(worker.current.left_detection, sharpness=float("nan")),
    )

    status = service.status()

    assert worker.snapshot_calls == 1
    assert status["session_id"] == "session-001"
    assert status["active_count"] == 0
    assert status["target_pairs"] == 4
    assert status["can_capture"] is True
    assert status["camera_error"] is None
    assert status["left"] == {
        "device": LEFT_DEVICE,
        "label": "cam-left",
        "rotation_degrees": 0,
        "health": "ready",
        "found": True,
        "saveable": True,
        "corner_count": 6,
        "sharpness": None,
        "saturation": 0.0125,
        "border_ok": True,
        "reasons": [],
    }
    assert status["right"]["rotation_degrees"] == 180
    assert "cameras" not in status
    assert status["calibration"] == {
        "state": "idle",
        "job_id": None,
        "report_path": None,
        "error": None,
    }
    json.dumps(status, allow_nan=False)
    assert str(tmp_path) not in json.dumps(status)


@pytest.mark.parametrize(
    "snapshot",
    [
        WorkerSnapshot(1, None, None, None, None),
        replace(good_snapshot(), error="camera disconnected"),
        replace(good_snapshot(), left_detection=detection(saveable=False, reasons=("blur",))),
        replace(good_snapshot(), right_detection=None),
    ],
)
def test_status_disables_capture_for_incomplete_unhealthy_or_unsaveable_pair(
    tmp_path: Path, snapshot: WorkerSnapshot
) -> None:
    service, _worker, _store, _config = make_service(tmp_path, snapshot)
    assert service.status()["can_capture"] is False


def test_status_hides_absolute_paths_in_worker_errors_and_quality_reasons(
    tmp_path: Path,
) -> None:
    private_path = f"{tmp_path}/private/camera.log"
    snapshot = replace(
        good_snapshot(),
        error=f"camera failed at {private_path}",
        left_detection=detection(saveable=False, reasons=(f"bad file {private_path}",)),
    )
    service, _worker, _store, _config = make_service(tmp_path, snapshot)

    encoded = json.dumps(service.status(), allow_nan=False)
    assert str(tmp_path) not in encoded
    assert "<path>" in encoded
    assert LEFT_DEVICE in encoded  # configured device identifiers are intentional display data


def test_capture_saves_exact_snapshot_images_and_audit_metadata(tmp_path: Path) -> None:
    service, worker, store, _config = make_service(tmp_path)
    expected = worker.current

    saved = service.capture()

    assert worker.snapshot_calls == 1
    assert saved.pair_id == 1
    left = cv2.imread(str(saved.left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(saved.right_path), cv2.IMREAD_COLOR)
    assert np.array_equal(left, expected.pair.left)
    assert np.array_equal(right, expected.pair.right)
    metadata = store.active_pairs()[0].metadata
    assert metadata["worker_sequence"] == 7
    assert metadata["timestamps_ns"] == {"left": 1_000_000, "right": 1_000_100}
    assert metadata["image_size"] == [160, 120]
    assert np.asarray(metadata["left_corners"]).shape == (6, 2)
    assert np.asarray(metadata["right_corners"]).shape == (6, 2)
    assert metadata["quality"]["left"]["sharpness"] == 123.25
    assert metadata["quality"]["right"]["reasons"] == []


def test_quarter_turn_service_uses_logical_dimensions_for_status_and_storage(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, left_rotation=270, right_rotation=90)
    logical_left = np.zeros((160, 120, 3), dtype=np.uint8)
    logical_right = np.zeros((160, 120, 3), dtype=np.uint8)
    snapshot = replace(
        good_snapshot(),
        pair=FramePair(logical_left, logical_right, 1_000_000, 1_000_100),
    )
    service, _worker, store, _config = make_service(
        tmp_path, snapshot, config=config
    )

    assert service.status()["can_capture"] is True
    service.capture()

    saved = store.active_pairs()[0]
    assert cv2.imread(str(saved.left_path)).shape == (160, 120, 3)
    assert saved.metadata["image_size"] == [120, 160]


def test_capture_rejects_duplicate_worker_sequence_without_second_write(tmp_path: Path) -> None:
    service, _worker, store, _config = make_service(tmp_path)
    service.capture()

    with pytest.raises(ServiceError, match="already captured"):
        service.capture()

    assert len(store.active_pairs()) == 1


@pytest.mark.parametrize(
    "bad_snapshot",
    [
        replace(good_snapshot(), left_detection=detection(saveable=False, reasons=("blur",))),
        replace(
            good_snapshot(),
            right_detection=detection(corners=good_corners()[:-1]),
        ),
        replace(
            good_snapshot(),
            left_detection=detection(corners=np.full((6, 1, 2), np.nan, np.float32)),
        ),
        replace(
            good_snapshot(),
            pair=FramePair(
                np.zeros((10, 10, 3), np.uint8),
                np.zeros((120, 160, 3), np.uint8),
                1,
                2,
            ),
        ),
    ],
)
def test_capture_rejects_invalid_or_unsaveable_snapshot_without_writing(
    tmp_path: Path, bad_snapshot: WorkerSnapshot
) -> None:
    service, _worker, store, _config = make_service(tmp_path, bad_snapshot)
    with pytest.raises(ServiceError):
        service.capture()
    assert store.active_pairs() == []


def test_capture_rejects_corners_outside_the_configured_image(tmp_path: Path) -> None:
    outside = good_corners()
    outside[0, 0] = (160.0, 30.0)
    snapshot = replace(
        good_snapshot(),
        left_detection=detection(corners=outside),
    )
    service, _worker, store, _config = make_service(tmp_path, snapshot)

    assert service.status()["can_capture"] is False
    with pytest.raises(ServiceError, match="left checkerboard corners.*image bounds"):
        service.capture()
    assert store.active_pairs() == []


def test_reject_last_and_retry_translate_operations_without_mutating_other_data(
    tmp_path: Path,
) -> None:
    service, worker, store, _config = make_service(tmp_path)
    service.capture()
    worker.current = replace(worker.current, sequence=8)
    service.capture()

    rejected = service.reject_last()
    remaining_before_retry = [pair.pair_id for pair in store.active_pairs()]
    service.retry_cameras()

    assert rejected.pair_id == 2
    assert remaining_before_retry == [1]
    assert [pair.pair_id for pair in store.active_pairs()] == [1]
    assert worker.retry_calls == 1
    manifest = store.manifest_path.read_text(encoding="utf-8")
    assert "operator requested deletion" in manifest


def test_reject_last_without_active_pair_is_service_error(tmp_path: Path) -> None:
    service, _worker, _store, _config = make_service(tmp_path)
    with pytest.raises(ServiceError, match="no active"):
        service.reject_last()


def add_active_pairs(store: SessionStore, count: int) -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    for index in range(count):
        corners = (good_corners().reshape(-1, 2) + index * 0.01).tolist()
        store.save_pair(
            image,
            image,
            {"image_size": [160, 120], "left_corners": corners, "right_corners": corners},
        )


def test_background_calibration_transitions_and_uses_exact_valid_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path, left_rotation=270, right_rotation=90)
    service, _worker, store, config = make_service(tmp_path, config=config)
    add_active_pairs(store, 4)
    entered = threading.Event()
    release = threading.Event()
    calls: dict[str, object] = {}

    observations = [SimpleNamespace(pair_id=value) for value in (1, 2, 3)]

    skipped = [{"pair_id": 4, "reason": "left_corners is missing"}]

    def fake_load(session, board):
        assert session is store
        assert board is config.checkerboard
        return observations, skipped

    def fake_calibrate(received, image_size, minimum_pairs):
        calls["calibrate"] = (received, image_size, minimum_pairs)
        entered.set()
        assert release.wait(2)
        return object()

    def fake_write(
        session, result, received_config, source_pairs, *, skip_diagnostics
    ):
        calls["source_ids"] = [pair.pair_id for pair in source_pairs]
        calls["skip_diagnostics"] = skip_diagnostics
        run = session.results_dir / "run-001"
        run.mkdir()
        (run / "report.json").write_text('{"status":"pass"}', encoding="utf-8")
        return run

    monkeypatch.setattr(service_module, "load_session_observations", fake_load)
    monkeypatch.setattr(service_module, "calibrate_observations", fake_calibrate)
    monkeypatch.setattr(service_module, "write_calibration_run", fake_write)

    job_id = service.start_calibration()
    assert entered.wait(2)
    running = service.status()["calibration"]
    assert job_id == "calibration-0001"
    assert running["state"] == "running"
    assert service._calibration_thread is not None
    assert service._calibration_thread.daemon is True
    with pytest.raises(ServiceError, match="already running"):
        service.start_calibration()
    release.set()

    completed = wait_for_state(service, "pass")["calibration"]
    assert calls["calibrate"] == (observations, (120, 160), 3)
    assert calls["source_ids"] == [1, 2, 3]
    assert calls["skip_diagnostics"] == skipped
    assert completed["report_path"] == "results/run-001/report.json"
    assert completed["error"] is None


def test_start_calibration_rejects_too_few_active_pairs_without_starting_job(
    tmp_path: Path,
) -> None:
    service, _worker, _store, _config = make_service(tmp_path)

    with pytest.raises(ServiceError, match="active pairs"):
        service.start_calibration()

    assert service._calibration_thread is None
    assert service.status()["calibration"]["state"] == "idle"


def test_thread_start_failure_becomes_stable_service_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _worker, store, _config = make_service(tmp_path)
    add_active_pairs(store, 3)

    class FailingThread:
        daemon = True

        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError(f"thread failed at {tmp_path}/private")

    monkeypatch.setattr(service_module.threading, "Thread", FailingThread)
    with pytest.raises(ServiceError, match="start calibration"):
        service.start_calibration()

    calibration = service.status()["calibration"]
    assert calibration["state"] == "error"
    assert str(tmp_path) not in calibration["error"]


def test_background_calibration_surfaces_skipped_pair_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _worker, store, _config = make_service(tmp_path)
    add_active_pairs(store, 3)
    monkeypatch.setattr(
        service_module,
        "load_session_observations",
        lambda *_args: (
            [SimpleNamespace(pair_id=1), SimpleNamespace(pair_id=2)],
            [{"pair_id": 3, "reason": "corner metadata invalid"}],
        ),
    )

    service.start_calibration()
    status = wait_for_state(service, "error")["calibration"]
    assert "2 valid pairs" in status["error"]
    assert "pair 3" in status["error"]
    assert "corner metadata invalid" in status["error"]


def test_background_calibration_error_is_stable_and_hides_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _worker, store, _config = make_service(tmp_path)
    add_active_pairs(store, 3)

    def fail(*_args):
        raise RuntimeError(f"failed at {tmp_path}/private/model.bin")

    monkeypatch.setattr(service_module, "load_session_observations", fail)
    service.start_calibration()
    status = wait_for_state(service, "error")["calibration"]
    assert status["error"].startswith("RuntimeError:")
    assert str(tmp_path) not in status["error"]
    assert status["report_path"] is None


def test_preview_uses_one_snapshot_and_decodes_to_side_by_side_canvas(tmp_path: Path) -> None:
    service, worker, _store, _config = make_service(tmp_path)
    original = np.hstack((worker.current.pair.left, worker.current.pair.right))

    encoded = service.preview_jpeg()
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert worker.snapshot_calls == 1
    assert isinstance(encoded, bytes)
    assert decoded.shape == (120, 320, 3)
    assert not np.array_equal(decoded, original)


def test_preview_requires_complete_pair(tmp_path: Path) -> None:
    service, _worker, _store, _config = make_service(
        tmp_path, WorkerSnapshot(0, None, None, None, "offline")
    )
    with pytest.raises(ServiceError, match="complete"):
        service.preview_jpeg()


def test_preview_draws_only_exact_board_shape_with_configured_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, worker, _store, _config = make_service(tmp_path)
    patterns: list[tuple[int, int]] = []

    def record_pattern(_image, pattern_size, _corners, _found):
        patterns.append(pattern_size)

    monkeypatch.setattr(cv2, "drawChessboardCorners", record_pattern)
    service.preview_jpeg()
    assert patterns == [(3, 2), (3, 2)]

    patterns.clear()
    worker.current = replace(
        worker.current,
        left_detection=detection(corners=good_corners()[:-1]),
    )
    service.preview_jpeg()
    assert patterns == [(3, 2)]


def test_preview_encode_failure_is_service_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _worker, _store, _config = make_service(tmp_path)
    monkeypatch.setattr(cv2, "imencode", lambda *_args: (False, None))
    with pytest.raises(ServiceError, match="^preview encoding failed$"):
        service.preview_jpeg()


class FakeHTTPService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.status_payload: object = {"ok": True}
        self.failure: Exception | None = None

    def _call(self, name: str):
        self.calls.append(name)
        if self.failure is not None:
            raise self.failure

    def status(self):
        self._call("status")
        return self.status_payload

    def preview_jpeg(self):
        self._call("preview")
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        return encoded.tobytes()

    def capture(self):
        self._call("capture")
        return SimpleNamespace(pair_id=12)

    def reject_last(self):
        self._call("reject")
        return SimpleNamespace(pair_id=11)

    def retry_cameras(self):
        self._call("retry")

    def start_calibration(self):
        self._call("calibrate")
        return "calibration-0001"


class RunningServer:
    def __init__(self, service: object) -> None:
        self.server = build_server("127.0.0.1", 0, service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            method=method,
            headers=headers or {},
            data=data,
        )
        try:
            response = urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items()), exc.read()
        with response:
            return response.status, dict(response.headers.items()), response.read()

    def duplicate_header_request(
        self, path: str, name: str, first: str, second: str
    ) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.putrequest("POST", path)
            connection.putheader(name, first)
            connection.putheader(name, second)
            connection.endheaders()
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()


def assert_security_headers(headers: dict[str, str], body: bytes) -> None:
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert int(headers["Content-Length"]) == len(body)


def test_server_refuses_non_loopback_bind_before_construction() -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        build_server("0.0.0.0", 0, FakeHTTPService())


def test_exact_get_routes_content_types_and_security_headers() -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        cases = [
            ("/", 200, "text/html; charset=utf-8"),
            ("/api/status", 200, "application/json; charset=utf-8"),
            ("/api/preview.jpg?cache=123", 200, "image/jpeg"),
            ("/missing", 404, "application/json; charset=utf-8"),
        ]
        for path, expected_status, content_type in cases:
            status, headers, body = running.request(path)
            assert status == expected_status
            assert headers["Content-Type"] == content_type
            assert_security_headers(headers, body)
    assert fake.calls == ["status", "preview"]


def test_exact_post_routes_and_status_codes() -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        origin = {"Origin": f"http://localhost:{running.port}"}
        assert running.request("/api/capture", method="POST", headers=origin)[0] == 201
        assert running.request("/api/reject-last", method="POST", headers=origin)[0] == 200
        assert running.request("/api/retry-cameras", method="POST", headers=origin)[0] == 200
        assert running.request("/api/calibrate", method="POST", headers=origin)[0] == 202
    assert fake.calls == ["capture", "reject", "retry", "calibrate"]


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "https://localhost:{port}",
        "http://localhost",
        "http://localhost:{port}/suffix",
        "http://localhost:{port}.evil.example",
        "http://user@localhost:{port}",
        "http://[::1]:{port}",
        "http://127.0.0.1:1",
    ],
)
def test_post_origin_validation_rejects_non_exact_values(origin: str) -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        status, headers, body = running.request(
            "/api/capture",
            method="POST",
            headers={"Origin": origin.format(port=running.port)},
        )
        assert status == 403
        assert_security_headers(headers, body)
    assert fake.calls == []


def test_post_allows_absent_origin_and_both_exact_loopback_origins() -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        for headers in (
            {},
            {"Origin": f"http://localhost:{running.port}"},
            {"Origin": f"http://127.0.0.1:{running.port}"},
        ):
            assert running.request("/api/retry-cameras", method="POST", headers=headers)[0] == 200
    assert fake.calls == ["retry", "retry", "retry"]


def test_post_rejects_duplicate_origin_and_content_length_headers() -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        allowed = f"http://localhost:{running.port}"
        assert running.duplicate_header_request(
            "/api/retry-cameras", "Origin", allowed, "http://evil.example"
        )[0] == 403
        assert running.duplicate_header_request(
            "/api/retry-cameras", "Content-Length", "0", "1"
        )[0] == 400
    assert fake.calls == []


@pytest.mark.parametrize(
    ("headers", "data", "expected"),
    [
        ({"Content-Length": "x"}, None, 400),
        ({"Content-Length": "1"}, None, 413),
        ({"Transfer-Encoding": "chunked"}, None, 400),
        ({}, b"payload", 413),
    ],
)
def test_post_rejects_malformed_or_nonempty_bodies_without_dispatch(
    headers: dict[str, str], data: bytes | None, expected: int
) -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        assert running.request(
            "/api/retry-cameras", method="POST", headers=headers, data=data
        )[0] == expected
    assert fake.calls == []


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/status", "POST"),
        ("/api/capture", "GET"),
        ("/", "POST"),
        ("/api/status", "HEAD"),
        ("/api/status", "PUT"),
        ("/api/capture", "DELETE"),
        ("/", "OPTIONS"),
        ("/api/status", "BREW"),
    ],
)
def test_known_route_with_wrong_method_returns_405(path: str, method: str) -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        status, headers, body = running.request(path, method=method)
        assert status == 405
        assert "Allow" in headers
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        if method != "HEAD":
            assert int(headers["Content-Length"]) == len(body)
        else:
            assert body == b""
            assert int(headers["Content-Length"]) > 0
    assert fake.calls == []


def test_service_conflict_maps_to_409_and_unexpected_error_is_sanitized() -> None:
    fake = FakeHTTPService()
    with RunningServer(fake) as running:
        fake.failure = ServiceError("capture conflict")
        status, _headers, body = running.request("/api/capture", method="POST")
        assert status == 409
        assert json.loads(body) == {"error": "capture conflict"}

        fake.failure = RuntimeError("secret /home/mzq/private/file.py")
        status, _headers, body = running.request("/api/status")
        assert status == 500
        assert json.loads(body) == {"error": "internal server error"}
        assert b"/home/mzq" not in body


def test_nonfinite_api_payload_becomes_controlled_500() -> None:
    fake = FakeHTTPService()
    fake.status_payload = {"bad": float("nan")}
    with RunningServer(fake) as running:
        status, headers, body = running.request("/api/status")
        assert status == 500
        assert json.loads(body) == {"error": "internal server error"}
        assert_security_headers(headers, body)


def test_two_concurrent_http_captures_save_only_one_pair(tmp_path: Path) -> None:
    service, _worker, store, _config = make_service(tmp_path)
    with RunningServer(service) as running:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(running.request, "/api/capture", method="POST")
                for _ in range(2)
            ]
            statuses = sorted(future.result()[0] for future in futures)
    assert statuses == [201, 409]
    assert len(store.active_pairs()) == 1


def test_http_unsaveable_capture_is_409_and_does_not_write(tmp_path: Path) -> None:
    snapshot = replace(
        good_snapshot(), left_detection=detection(saveable=False, reasons=("blur",))
    )
    service, _worker, store, _config = make_service(tmp_path, snapshot)
    with RunningServer(service) as running:
        status, _headers, body = running.request("/api/capture", method="POST")
    assert status == 409
    assert "not saveable" in json.loads(body)["error"]
    assert store.active_pairs() == []


def test_http_service_error_does_not_expose_worker_absolute_path(tmp_path: Path) -> None:
    snapshot = replace(good_snapshot(), error=f"failed at {tmp_path}/private/device.log")
    service, _worker, store, _config = make_service(tmp_path, snapshot)
    with RunningServer(service) as running:
        status, _headers, body = running.request("/api/capture", method="POST")
    assert status == 409
    assert str(tmp_path).encode() not in body
    assert b"<path>" in body
    assert store.active_pairs() == []


def test_index_is_self_contained_safe_chinese_guided_interface() -> None:
    index_path = Path(service_module.__file__).with_name("static") / "index.html"
    text = index_path.read_text(encoding="utf-8")

    for required in (
        "保存这一组",
        "移除上一组",
        "重新连接相机",
        "开始标定",
        "标定板停止移动后再保存",
        "中心",
        "四个角",
        "相反方向倾斜",
        "近距离",
        "远距离",
    ):
        assert required in text
    assert "textContent" in text
    assert "innerHTML" not in text
    assert "http://" not in text
    assert "https://" not in text
    assert "fetch('/" not in text
    assert "api/status" in text
    assert "api/preview.jpg" in text
    assert "200" in text
    assert "new Image" not in text  # one reusable preview element, no piling image objects
