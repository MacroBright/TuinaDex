"""Hardware-free tests for the stereo calibration command-line lifecycle."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.stereo_calibration.cameras import CameraError, FramePair, WorkerSnapshot
from tools.stereo_calibration.config import AppConfig, ConfigError
from tools.stereo_calibration.main import (
    build_parser,
    main,
    prepare_session,
    run_camera_check,
    serve_session,
)
from tools.stereo_calibration.tests.test_config import valid_payload


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_mapping(valid_payload(tmp_path))


def test_cli_defaults_to_example_config() -> None:
    args = build_parser().parse_args([])
    assert args.config.name == "example_config.json"
    assert args.session is None
    assert args.check_cameras is False


def test_cli_can_resume_named_session_and_check_cameras() -> None:
    args = build_parser().parse_args(
        ["--session", "20260826-dry-run", "--check-cameras"]
    )
    assert args.session == "20260826-dry-run"
    assert args.check_cameras is True


def test_new_session_writes_exact_configuration_snapshot(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "20260826-dry-run")

    snapshot = json.loads(
        (store.session_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot == config.to_mapping()
    assert not (store.session_dir / "config_snapshot.tmp").exists()


def test_resume_preserves_pairs_and_accepts_exact_configuration(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "20260826-dry-run")
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    store.save_pair(image, image, {"marker": "kept"})

    reopened = prepare_session(config, "20260826-dry-run")

    assert [pair.pair_id for pair in reopened.active_pairs()] == [1]
    assert reopened.active_pairs()[0].metadata == {"marker": "kept"}


def test_resume_refuses_changed_effective_configuration(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    session = prepare_session(config, "20260826-dry-run")
    changed = replace(config, baseline_reference_mm=config.baseline_reference_mm + 10.0)

    with pytest.raises(ConfigError, match="does not match the session snapshot"):
        prepare_session(changed, session.session_dir.name)


@pytest.mark.parametrize("contents", [None, "not json", "[]", '{"bad": NaN}'])
def test_resume_refuses_missing_or_invalid_snapshot(
    tmp_path: Path, contents: str | None
) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "broken-resume")
    snapshot = store.session_dir / "config_snapshot.json"
    snapshot.unlink()
    if contents is not None:
        snapshot.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match="snapshot"):
        prepare_session(config, "broken-resume")


def test_resume_refuses_symlinked_snapshot(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "symlink-resume")
    snapshot = store.session_dir / "config_snapshot.json"
    external = tmp_path / "external.json"
    external.write_text(json.dumps(config.to_mapping()), encoding="utf-8")
    snapshot.unlink()
    snapshot.symlink_to(external)

    with pytest.raises(ConfigError, match="snapshot"):
        prepare_session(config, "symlink-resume")


@pytest.mark.parametrize("session_id", ["", ".", "..", "../escape", "nested/name"])
def test_prepare_session_rejects_unsafe_id_before_path_access(
    tmp_path: Path, session_id: str
) -> None:
    config = make_config(tmp_path)

    with pytest.raises(ConfigError, match="safe single path component"):
        prepare_session(config, session_id)

    assert not config.data_root.exists()


class FakeCameraPair:
    def __init__(self, *, read_error: Exception | None = None) -> None:
        self.open_count = 0
        self.read_count = 0
        self.close_count = 0
        self.read_error = read_error

    def open(self) -> None:
        self.open_count += 1

    def read(self) -> FramePair:
        self.read_count += 1
        if self.read_error is not None:
            raise self.read_error
        image = np.zeros((960, 1280, 3), dtype=np.uint8)
        return FramePair(image, image, 10, 11)

    def close(self) -> None:
        self.close_count += 1


def test_camera_check_reads_one_pair_and_always_closes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    pair = FakeCameraPair()

    captured = run_camera_check(config, pair_factory=lambda _config: pair)

    assert captured.left.shape == (960, 1280, 3)
    assert (pair.open_count, pair.read_count, pair.close_count) == (1, 1, 1)


def test_camera_check_preserves_read_error_and_closes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    pair = FakeCameraPair(read_error=CameraError("read failed"))

    with pytest.raises(CameraError, match="read failed"):
        run_camera_check(config, pair_factory=lambda _config: pair)

    assert pair.close_count == 1


class FakeWorker:
    def __init__(self, snapshot: WorkerSnapshot) -> None:
        self.current = snapshot
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    def _wait_for_snapshot(self, predicate, timeout: float) -> WorkerSnapshot:
        assert timeout > 0
        assert predicate(self.current)
        return self.current

    def snapshot(self) -> WorkerSnapshot:
        return self.current

    def retry(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_count += 1


class FakeServer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.server_address = ("127.0.0.1", 8765)

    def serve_forever(self, *, poll_interval: float) -> None:
        assert 0 < poll_interval <= 0.5
        self.events.append("serve")
        raise KeyboardInterrupt

    def server_close(self) -> None:
        self.events.append("server_close")


def ready_snapshot() -> WorkerSnapshot:
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    pair = FramePair(image, image, 100, 101)
    return WorkerSnapshot(1, pair, None, None, None)


def test_serve_session_prints_mapping_and_closes_server_then_worker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "serve-test")
    worker = FakeWorker(ready_snapshot())
    events: list[str] = []

    def server_factory(host, port, service):
        assert (host, port) == ("127.0.0.1", 8765)
        assert service.status()["session_id"] == "serve-test"
        return FakeServer(events)

    exit_code = serve_session(config, store, worker, server_factory=server_factory)

    assert exit_code == 0
    assert worker.start_count == worker.stop_count == 1
    assert events == ["serve", "server_close"]
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8765" in output
    assert "ssh -N -L 8765:127.0.0.1:8765 mzq@mzq.local" in output
    assert str(store.session_dir) in output
    assert config.left.device in output and config.right.device in output
    assert "180" in output


def test_serve_session_rejects_initial_camera_error_and_stops_worker(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "camera-error")
    worker = FakeWorker(WorkerSnapshot(1, None, None, None, "offline"))

    with pytest.raises(CameraError, match="offline"):
        serve_session(
            config,
            store,
            worker,
            server_factory=lambda *_args: pytest.fail("server must not start"),
        )

    assert worker.stop_count == 1


def test_serve_session_attempts_all_cleanup_and_preserves_primary_failure(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    store = prepare_session(config, "cleanup-errors")
    worker = FakeWorker(ready_snapshot())

    def bad_stop() -> None:
        worker.stop_count += 1
        raise RuntimeError("worker cleanup boom")

    worker.stop = bad_stop

    class FailingServer(FakeServer):
        def serve_forever(self, *, poll_interval: float) -> None:
            raise RuntimeError("serve boom")

        def server_close(self) -> None:
            self.events.append("server_close")
            raise RuntimeError("server cleanup boom")

    events: list[str] = []
    with pytest.raises(RuntimeError, match="server cleanup boom.*worker cleanup boom") as info:
        serve_session(
            config,
            store,
            worker,
            server_factory=lambda *_args: FailingServer(events),
        )

    assert isinstance(info.value.__cause__, RuntimeError)
    assert str(info.value.__cause__) == "serve boom"
    assert events == ["server_close"]
    assert worker.stop_count == 1


def test_check_cameras_mode_does_not_create_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = valid_payload(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    called: list[AppConfig] = []

    monkeypatch.setattr(
        "tools.stereo_calibration.main.run_camera_check",
        lambda config: called.append(config) or ready_snapshot().pair,
    )

    assert main(["--config", str(config_path), "--check-cameras"]) == 0
    assert len(called) == 1
    assert not Path(payload["data_root"]).exists()


def test_main_returns_nonzero_for_configuration_error(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.json"
    assert main(["--config", str(missing)]) != 0
    assert "错误" in capsys.readouterr().err


def test_beginner_readme_contains_the_complete_operation_boundary() -> None:
    readme = Path(__file__).parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    required_sections = [
        "这个工具做什么、不做什么",
        "实物检查清单",
        "Ubuntu 环境检查",
        "相机预检",
        "启动命令与预期输出",
        "Mac SSH 隧道",
        "三组试采",
        "正式 30 组姿态清单",
        "质量阈值与报告解读",
        "恢复、重连与安全退出",
        "什么时候必须重新标定",
        "USB 非硬同步局限",
    ]
    assert all(section in text for section in required_sections)
    assert "source /home/mzq/miniconda3/etc/profile.d/conda.sh" in text
    assert "conda activate tuinadex_hw" in text
    assert "mzq@mzq.local" in text
    assert "--check-cameras" in text
    assert "54/54" in text
    assert "suspected_outliers" in text and "skipped_pairs" in text
    assert "点云" in text and "LeRobot" in text
