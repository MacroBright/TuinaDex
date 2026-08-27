"""Command-line lifecycle for guided stereo camera calibration."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import signal
import stat
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .cameras import CameraError, CameraWorker, FramePair, OpenCVCameraPair
from .config import AppConfig, ConfigError
from .service import CalibrationService
from .session import SessionError, SessionStore
from .web import build_server


class _ShutdownRequested(Exception):
    """Internal control-flow exception raised by SIGINT and SIGTERM."""


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def build_parser() -> argparse.ArgumentParser:
    """Return the testable command-line parser."""
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Guided stereo camera calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=package_dir / "example_config.json",
        help="validated JSON configuration",
    )
    parser.add_argument(
        "--session",
        help="safe session identifier to create or resume; default creates a timestamp",
    )
    parser.add_argument(
        "--check-cameras",
        action="store_true",
        help="open both cameras, read one pair, close them, and exit",
    )
    return parser


def prepare_session(config: AppConfig, session_id: str | None) -> SessionStore:
    """Create or resume a session whose effective configuration is identical."""
    effective = config.to_mapping()
    if session_id is None:
        return _create_timestamped_session(config, effective)
    if (
        not isinstance(session_id, str)
        or session_id in ("", ".", "..")
        or _SAFE_SESSION_ID.fullmatch(session_id) is None
    ):
        raise ConfigError("session ID must be a safe single path component")

    session_dir = config.data_root / session_id
    if session_dir.is_symlink():
        raise ConfigError("existing session directory must not be a symlink")
    if session_dir.exists():
        _require_matching_snapshot(session_dir / "config_snapshot.json", effective)
        try:
            return SessionStore.open(session_dir)
        except SessionError as exc:
            raise ConfigError("existing session storage is invalid") from exc
    return _create_session_with_snapshot(config, session_id, effective)


def _create_timestamped_session(
    config: AppConfig, effective: dict[str, Any]
) -> SessionStore:
    timestamp = datetime.now()
    for _attempt in range(1_000):
        session_id = timestamp.strftime("%Y%m%d-%H%M%S%f")
        if not (config.data_root / session_id).exists():
            try:
                return _create_session_with_snapshot(config, session_id, effective)
            except SessionError as exc:
                if "already exists" not in str(exc):
                    raise
        timestamp += timedelta(microseconds=1)
    raise ConfigError("could not allocate a unique timestamped session")


def _create_session_with_snapshot(
    config: AppConfig, session_id: str, effective: dict[str, Any]
) -> SessionStore:
    try:
        store = SessionStore.create(config.data_root, session_id)
    except SessionError:
        raise
    _write_configuration_snapshot(store.session_dir, effective)
    return store


def _write_configuration_snapshot(
    session_dir: Path, effective: dict[str, Any]
) -> None:
    snapshot_path = session_dir / "config_snapshot.json"
    temporary = snapshot_path.with_suffix(".tmp")
    if snapshot_path.is_symlink() or snapshot_path.exists() or temporary.is_symlink():
        raise ConfigError("configuration snapshot path is not empty")
    payload = (
        json.dumps(effective, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ConfigError("configuration snapshot temporary is not a regular file")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("configuration snapshot write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, snapshot_path)
        _fsync_directory(session_dir)
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        raise ConfigError("could not write configuration snapshot") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass


def _require_matching_snapshot(
    snapshot_path: Path, effective: dict[str, Any]
) -> None:
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ConfigError("existing session has no valid configuration snapshot")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        saved = json.loads(
            snapshot_path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigError("existing session configuration snapshot is invalid") from exc
    if not isinstance(saved, dict):
        raise ConfigError("existing session configuration snapshot is invalid")
    if saved != effective:
        raise ConfigError("configuration does not match the session snapshot")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            unsupported = {
                errno.EINVAL,
                errno.ENOSYS,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def run_camera_check(
    config: AppConfig,
    *,
    pair_factory: Callable[[AppConfig], Any] = OpenCVCameraPair,
) -> FramePair:
    """Open, validate, and close one hardware pair without creating a session."""
    pair = pair_factory(config)
    primary: BaseException | None = None
    captured: FramePair | None = None
    try:
        pair.open()
        captured = pair.read()
        if not isinstance(captured, FramePair):
            raise CameraError("camera check did not return a FramePair")
    except BaseException as exc:
        primary = exc
    close_error: BaseException | None = None
    try:
        pair.close()
    except BaseException as exc:
        close_error = exc
    if primary is not None:
        if close_error is not None:
            raise CameraError(
                f"camera check failed: {primary}; camera close failed: {close_error}"
            ) from primary
        raise primary
    if close_error is not None:
        raise CameraError(f"camera close failed: {close_error}") from close_error
    assert captured is not None
    return captured


def serve_session(
    config: AppConfig,
    session: SessionStore,
    worker: Any,
    *,
    server_factory: Callable[[str, int, Any], Any] = build_server,
    initial_timeout: float = 10.0,
) -> int:
    """Run one session until a signal/interrupt, then close server and cameras."""
    server: Any | None = None
    worker_started = False
    primary: BaseException | None = None
    result = 0
    try:
        with _shutdown_signals():
            worker_started = True
            worker.start()
            snapshot = worker._wait_for_snapshot(
                lambda item: item.error is not None or item.pair is not None,
                timeout=initial_timeout,
            )
            if snapshot.error is not None:
                raise CameraError(f"initial camera error: {snapshot.error}")
            if snapshot.pair is None:
                raise CameraError(
                    f"cameras did not produce a complete pair within {initial_timeout:g} seconds"
                )
            service = CalibrationService(worker, session, config)
            server = server_factory(config.web.host, config.web.port, service)
            selected_port = int(server.server_address[1])
            _print_startup(config, session, selected_port)
            server.serve_forever(poll_interval=0.2)
    except (KeyboardInterrupt, _ShutdownRequested):
        result = 0
    except BaseException as exc:
        primary = exc

    cleanup_errors: list[str] = []
    if server is not None:
        try:
            server.server_close()
        except BaseException as exc:
            cleanup_errors.append(f"web server close failed: {exc}")
    if worker_started:
        try:
            worker.stop()
        except BaseException as exc:
            cleanup_errors.append(f"camera worker stop failed: {exc}")

    if primary is not None:
        if cleanup_errors:
            raise RuntimeError(
                f"application failed: {primary}; cleanup errors: {'; '.join(cleanup_errors)}"
            ) from primary
        raise primary
    if cleanup_errors:
        raise RuntimeError("; ".join(cleanup_errors))
    return result


@contextmanager
def _shutdown_signals() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise _ShutdownRequested

    previous: dict[signal.Signals, Any] = {}
    for selected in (signal.SIGINT, signal.SIGTERM):
        previous[selected] = signal.getsignal(selected)
        signal.signal(selected, request_shutdown)
    try:
        yield
    finally:
        for selected, handler in previous.items():
            signal.signal(selected, handler)


def _print_startup(config: AppConfig, session: SessionStore, port: int) -> None:
    print(f"Ubuntu 访问地址：http://127.0.0.1:{port}")
    print(
        "Mac SSH 隧道："
        f"ssh -N -L {port}:127.0.0.1:{port} mzq@mzq.local"
    )
    print(f"会话目录：{session.session_dir}")
    print(
        f"左相机：{config.left.name} -> {config.left.device} "
        f"(旋转 {config.left.rotation_degrees}°)"
    )
    print(
        f"右相机：{config.right.name} -> {config.right.device} "
        f"(旋转 {config.right.rotation_degrees}°)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        config = AppConfig.from_json(args.config)
        if args.check_cameras:
            pair = run_camera_check(config)
            print(
                "相机检查通过："
                f"{pair.left.shape[1]}x{pair.left.shape[0]}, "
                f"右相机旋转 {config.right.rotation_degrees}°"
            )
            return 0
        session = prepare_session(config, args.session)
        worker = CameraWorker(
            lambda: OpenCVCameraPair(config),
            config.checkerboard,
            config.quality,
        )
        return serve_session(config, session, worker)
    except (ConfigError, SessionError, CameraError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"错误：{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
