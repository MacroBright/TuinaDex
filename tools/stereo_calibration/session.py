"""Durable, resumable storage for stereo calibration image pairs."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_PAIR_FILE = re.compile(r"^pair_(\d+)_(left|right)\.png$")
_LOCK_FILENAME = ".session.lock"
_REJECT_TRANSACTION_FILENAME = ".reject-transaction.json"
_REJECT_TRANSACTION_TEMP_FILENAME = ".reject-transaction.tmp"
_IN_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_IN_PROCESS_LOCKS_GUARD = threading.Lock()


class SessionError(RuntimeError):
    """Raised when stereo capture session storage is invalid or cannot be updated."""


@dataclass(frozen=True)
class SavedPair:
    """One saved left/right stereo image pair."""

    pair_id: int
    left_path: Path
    right_path: Path
    metadata: dict[str, Any]


class SessionStore:
    """Append-only session manifest and corresponding pair image storage."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.pairs_dir = session_dir / "pairs"
        self.rejected_dir = session_dir / "rejected"
        self.results_dir = session_dir / "results"
        self.manifest_path = session_dir / "manifest.jsonl"
        self.lock_path = session_dir / _LOCK_FILENAME
        self.reject_transaction_path = session_dir / _REJECT_TRANSACTION_FILENAME
        self.reject_transaction_temp_path = session_dir / _REJECT_TRANSACTION_TEMP_FILENAME

    @classmethod
    def create(cls, data_root: Path, session_id: str) -> "SessionStore":
        """Create a new empty capture session beneath *data_root*."""
        if (
            not isinstance(session_id, str)
            or session_id in ("", ".", "..")
            or _SAFE_SESSION_ID.fullmatch(session_id) is None
        ):
            raise SessionError("session ID must be a safe single path component")

        session_dir = Path(data_root) / session_id
        if _path_exists_or_symlink(session_dir):
            raise SessionError(f"session already exists: {session_dir}")
        try:
            session_dir.mkdir(parents=True)
            (session_dir / "pairs").mkdir()
            (session_dir / "rejected").mkdir()
            (session_dir / "results").mkdir()
            (session_dir / "manifest.jsonl").touch(exist_ok=False)
        except OSError as exc:
            raise SessionError(f"could not create session {session_dir}") from exc

        store = cls(session_dir)
        lock_fd = store._open_lock_file()
        os.close(lock_fd)
        return store

    @classmethod
    def open(cls, session_dir: Path) -> "SessionStore":
        """Open, recover, and validate an existing capture session."""
        store = cls(Path(session_dir))
        with store._locked():
            store._replay_manifest()
        return store

    def save_pair(
        self,
        left: NDArray[np.uint8],
        right: NDArray[np.uint8],
        metadata: dict[str, Any],
    ) -> SavedPair:
        """Encode, publish, and durably log one stereo pair."""
        safe_metadata = _copy_metadata(metadata)
        with self._locked():
            events, _active = self._replay_manifest()
            pair_id = self._next_pair_id(events)
            left_path, right_path = self._pair_paths(self.pairs_dir, pair_id)
            left_tmp = _temporary_path(left_path)
            right_tmp = _temporary_path(right_path)
            targets = (left_path, right_path, left_tmp, right_tmp)
            if any(_path_exists_or_symlink(path) for path in targets):
                raise SessionError(f"refusing to overwrite files for pair {pair_id}")

            try:
                if not cv2.imwrite(str(left_tmp), left):
                    raise SessionError(f"could not encode left image for pair {pair_id}")
                if not cv2.imwrite(str(right_tmp), right):
                    raise SessionError(f"could not encode right image for pair {pair_id}")
                left_tmp.replace(left_path)
                right_tmp.replace(right_path)
                self._require_image_pair(self.pairs_dir, pair_id, "active", None)
                self._append_event(
                    {"action": "capture", "pair_id": pair_id, "metadata": safe_metadata}
                )
            except SessionError:
                raise
            except (cv2.error, OSError) as exc:
                raise SessionError(f"could not save pair {pair_id}") from exc
            finally:
                self._cleanup_temps(left_tmp, right_tmp)
            return SavedPair(pair_id, left_path, right_path, _copy_metadata(safe_metadata))

    def active_pairs(self) -> list[SavedPair]:
        """Return validated active pairs ordered by pair ID."""
        with self._locked():
            _events, active = self._replay_manifest()
            return [
                SavedPair(
                    pair_id,
                    *self._pair_paths(self.pairs_dir, pair_id),
                    _copy_metadata(metadata),
                )
                for pair_id, metadata in sorted(active.items())
            ]

    def reject_last(self, reason: str) -> SavedPair:
        """Move and durably mark the highest-ID active pair as rejected."""
        safe_reason = _copy_reason(reason)
        with self._locked():
            _events, active = self._replay_manifest()
            if not active:
                raise SessionError("no active pair to reject")
            pair_id = max(active)
            metadata = active[pair_id]
            self._write_reject_transaction(pair_id, safe_reason)
            try:
                self._complete_reject_transaction(pair_id, safe_reason)
            except SessionError:
                raise
            except OSError as exc:
                raise SessionError(f"could not reject pair {pair_id}") from exc
            left_path, right_path = self._pair_paths(self.rejected_dir, pair_id)
            return SavedPair(pair_id, left_path, right_path, _copy_metadata(metadata))

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Take the shared thread and advisory process locks for this session."""
        with _thread_lock_for(self.session_dir):
            self._require_layout()
            lock_fd = self._open_lock_file()
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                self._require_layout()
                self._recover_reject_transaction()
                yield
            except OSError as exc:
                raise SessionError(f"could not lock session {self.session_dir}") from exc
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def _require_layout(self) -> None:
        _require_directory(self.session_dir, "session directory")
        _require_directory(self.pairs_dir, "pairs directory")
        _require_directory(self.rejected_dir, "rejected directory")
        _require_directory(self.results_dir, "results directory")
        _require_regular_file(self.manifest_path, "manifest")

    def _open_lock_file(self) -> int:
        if self.lock_path.is_symlink():
            raise SessionError(f"session lock is a symlink: {self.lock_path}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise SessionError(f"could not open session lock {self.lock_path}") from exc
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise SessionError(f"session lock is not a regular file: {self.lock_path}")
            if self.lock_path.is_symlink():
                raise SessionError(f"session lock is a symlink: {self.lock_path}")
        except BaseException:
            os.close(lock_fd)
            raise
        return lock_fd

    def _replay_manifest(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        events, active, active_lines, rejected_lines = self._read_manifest()
        for pair_id, line_number in active_lines.items():
            self._require_image_pair(self.pairs_dir, pair_id, "active", line_number)
        for pair_id, line_number in rejected_lines.items():
            self._require_image_pair(self.rejected_dir, pair_id, "rejected", line_number)
        return events, active

    def _read_manifest(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        dict[int, dict[str, Any]],
        dict[int, int],
        dict[int, int],
    ]:
        events: list[dict[str, Any]] = []
        active: dict[int, dict[str, Any]] = {}
        active_lines: dict[int, int] = {}
        rejected_lines: dict[int, int] = {}
        captured: set[int] = set()
        manifest_fd = _open_existing_regular_file(self.manifest_path, os.O_RDONLY, "manifest")
        try:
            with os.fdopen(manifest_fd, "r", encoding="utf-8") as manifest:
                for line_number, line in enumerate(manifest, start=1):
                    event = self._parse_event(line, line_number)
                    pair_id = event["pair_id"]
                    if event["action"] == "capture":
                        if pair_id in captured:
                            raise SessionError(
                                f"manifest line {line_number}: duplicate capture for pair {pair_id}"
                            )
                        captured.add(pair_id)
                        active[pair_id] = event["metadata"]
                        active_lines[pair_id] = line_number
                    else:
                        if pair_id not in active:
                            raise SessionError(
                                f"manifest line {line_number}: no active pair {pair_id} to reject"
                            )
                        del active[pair_id]
                        del active_lines[pair_id]
                        rejected_lines[pair_id] = line_number
                    events.append(event)
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"could not read manifest {self.manifest_path}") from exc
        return events, active, active_lines, rejected_lines

    @staticmethod
    def _parse_event(line: str, line_number: int) -> dict[str, Any]:
        try:
            event = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SessionError(f"manifest line {line_number}: malformed JSON") from exc
        if not isinstance(event, dict):
            raise SessionError(f"manifest line {line_number}: event must be an object")
        action = event.get("action")
        if action not in ("capture", "reject"):
            raise SessionError(f"manifest line {line_number}: action must be capture or reject")
        pair_id = event.get("pair_id")
        if type(pair_id) is not int or pair_id <= 0:
            raise SessionError(f"manifest line {line_number}: pair ID must be a positive integer")
        if action == "capture":
            if not isinstance(event.get("metadata"), dict):
                raise SessionError(
                    f"manifest line {line_number}: capture metadata must be an object"
                )
            try:
                event["metadata"] = _copy_metadata(event["metadata"])
            except SessionError as exc:
                raise SessionError(
                    f"manifest line {line_number}: pair {pair_id} has invalid capture metadata"
                ) from exc
        else:
            try:
                event["reason"] = _copy_reason(event.get("reason"))
            except SessionError as exc:
                raise SessionError(
                    f"manifest line {line_number}: pair {pair_id} has invalid reject reason"
                ) from exc
        return event

    def _require_image_pair(
        self, directory: Path, pair_id: int, state: str, line_number: int | None
    ) -> None:
        left_path, right_path = self._pair_paths(directory, pair_id)
        for path in (left_path, right_path):
            if path.is_symlink():
                raise SessionError(
                    f"{_line_prefix(line_number)}{state} pair {pair_id} image is a symlink"
                )
        left_state = _lstat_or_none(left_path)
        right_state = _lstat_or_none(right_path)
        if (left_state is None) != (right_state is None):
            raise SessionError(
                f"{_line_prefix(line_number)}{state} pair {pair_id} has mismatched left/right "
                f"image files (one is missing {state} image)"
            )
        if (
            left_state is None
            or right_state is None
            or not stat.S_ISREG(left_state.st_mode)
            or not stat.S_ISREG(right_state.st_mode)
        ):
            raise SessionError(
                f"{_line_prefix(line_number)}{state} pair {pair_id} is missing {state} image files"
            )

    def _next_pair_id(self, events: list[dict[str, Any]]) -> int:
        maximum = max((event["pair_id"] for event in events), default=0)
        for directory in (self.pairs_dir, self.rejected_dir):
            try:
                for candidate in directory.iterdir():
                    match = _PAIR_FILE.fullmatch(candidate.name)
                    candidate_state = _lstat_or_none(candidate)
                    if (
                        match
                        and candidate_state is not None
                        and stat.S_ISREG(candidate_state.st_mode)
                    ):
                        maximum = max(maximum, int(match.group(1)))
            except OSError as exc:
                raise SessionError(f"could not inspect session images in {directory}") from exc
        return maximum + 1

    @staticmethod
    def _pair_paths(directory: Path, pair_id: int) -> tuple[Path, Path]:
        stem = f"pair_{pair_id:04d}"
        return directory / f"{stem}_left.png", directory / f"{stem}_right.png"

    def _write_reject_transaction(self, pair_id: int, reason: str) -> None:
        if _path_exists_or_symlink(self.reject_transaction_path):
            raise SessionError("reject transaction marker already exists")
        if _path_exists_or_symlink(self.reject_transaction_temp_path):
            raise SessionError("reject transaction temporary marker already exists")
        payload = _compact_json({"pair_id": pair_id, "reason": reason}).encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            marker_fd = os.open(self.reject_transaction_temp_path, flags, 0o600)
            try:
                _write_all(marker_fd, payload)
                os.fsync(marker_fd)
            finally:
                os.close(marker_fd)
            os.replace(self.reject_transaction_temp_path, self.reject_transaction_path)
            _fsync_directory(self.session_dir)
        except OSError as exc:
            raise SessionError("could not write reject transaction marker") from exc

    def _recover_reject_transaction(self) -> None:
        transaction = self._read_reject_transaction()
        if transaction is not None:
            self._complete_reject_transaction(*transaction)

    def _read_reject_transaction(self) -> tuple[int, str] | None:
        marker_state = _lstat_or_none(self.reject_transaction_path)
        if marker_state is None:
            return None
        if stat.S_ISLNK(marker_state.st_mode):
            raise SessionError("reject transaction marker is a symlink")
        if not stat.S_ISREG(marker_state.st_mode):
            raise SessionError("reject transaction marker is not a regular file")
        marker_fd = _open_existing_regular_file(
            self.reject_transaction_path, os.O_RDONLY, "reject transaction marker"
        )
        try:
            with os.fdopen(marker_fd, "r", encoding="utf-8") as marker:
                payload = json.load(marker, parse_constant=_reject_json_constant)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise SessionError("could not read reject transaction marker") from exc
        if not isinstance(payload, dict):
            raise SessionError("reject transaction marker must be an object")
        pair_id = payload.get("pair_id")
        if type(pair_id) is not int or pair_id <= 0:
            raise SessionError("reject transaction marker has invalid pair ID")
        try:
            reason = _copy_reason(payload.get("reason"))
        except SessionError as exc:
            raise SessionError("reject transaction marker has invalid reason") from exc
        return pair_id, reason

    def _complete_reject_transaction(self, pair_id: int, reason: str) -> None:
        events, active, _active_lines, _rejected_lines = self._read_manifest()
        rejection_events = [
            event
            for event in events
            if event["action"] == "reject" and event["pair_id"] == pair_id
        ]
        if rejection_events and rejection_events[0]["reason"] != reason:
            raise SessionError(f"reject transaction reason disagrees with manifest for pair {pair_id}")
        if not rejection_events and pair_id not in active:
            raise SessionError(f"reject transaction has no active pair {pair_id}")
        self._move_reject_images(pair_id)
        if not rejection_events:
            self._append_event({"action": "reject", "pair_id": pair_id, "reason": reason})
        self._remove_reject_transaction()

    def _move_reject_images(self, pair_id: int) -> None:
        source_paths = self._pair_paths(self.pairs_dir, pair_id)
        target_paths = self._pair_paths(self.rejected_dir, pair_id)
        for source, target in zip(source_paths, target_paths):
            self._move_reject_image(pair_id, source, target)
        self._require_image_pair(self.rejected_dir, pair_id, "rejected", None)
        _fsync_directory(self.pairs_dir)
        _fsync_directory(self.rejected_dir)

    @staticmethod
    def _move_reject_image(pair_id: int, source: Path, target: Path) -> None:
        for path in (source, target):
            if path.is_symlink():
                raise SessionError(f"reject pair {pair_id} image is a symlink: {path}")
        source_state = _lstat_or_none(source)
        target_state = _lstat_or_none(target)
        if source_state is not None and not stat.S_ISREG(source_state.st_mode):
            raise SessionError(f"reject pair {pair_id} source is not a regular file")
        if target_state is not None and not stat.S_ISREG(target_state.st_mode):
            raise SessionError(f"reject pair {pair_id} target is not a regular file")
        if source_state is not None and target_state is not None:
            raise SessionError(f"reject pair {pair_id} has colliding source and rejected files")
        if source_state is None and target_state is None:
            raise SessionError(f"reject pair {pair_id} has both source and rejected files missing")
        if source_state is not None:
            source.replace(target)

    def _remove_reject_transaction(self) -> None:
        marker_state = _lstat_or_none(self.reject_transaction_path)
        if marker_state is None:
            raise SessionError("reject transaction marker disappeared")
        if stat.S_ISLNK(marker_state.st_mode):
            raise SessionError("reject transaction marker is a symlink")
        if not stat.S_ISREG(marker_state.st_mode):
            raise SessionError("reject transaction marker is not a regular file")
        try:
            self.reject_transaction_path.unlink()
            _fsync_directory(self.session_dir)
        except OSError as exc:
            raise SessionError("could not remove reject transaction marker") from exc

    def _append_event(self, event: dict[str, Any]) -> None:
        try:
            payload = _compact_json(event).encode("utf-8") + b"\n"
            manifest_fd = _open_existing_regular_file(
                self.manifest_path, os.O_WRONLY | os.O_APPEND, "manifest"
            )
            try:
                _write_all(manifest_fd, payload)
                os.fsync(manifest_fd)
            finally:
                os.close(manifest_fd)
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            raise SessionError("could not append manifest event") from exc

    @staticmethod
    def _cleanup_temps(*paths: Path) -> None:
        for path in paths:
            try:
                path_state = _lstat_or_none(path)
                if path_state is not None and stat.S_ISREG(path_state.st_mode):
                    path.unlink()
            except OSError:
                pass


def _thread_lock_for(session_dir: Path) -> threading.RLock:
    key = str(session_dir.absolute())
    with _IN_PROCESS_LOCKS_GUARD:
        return _IN_PROCESS_LOCKS.setdefault(key, threading.RLock())


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SessionError(f"could not inspect path {path}") from exc


def _path_exists_or_symlink(path: Path) -> bool:
    return _lstat_or_none(path) is not None


def _require_directory(path: Path, label: str) -> None:
    path_state = _lstat_or_none(path)
    if path_state is None:
        raise SessionError(f"session is missing expected {label}: {path}")
    if stat.S_ISLNK(path_state.st_mode):
        raise SessionError(f"{label} is a symlink: {path}")
    if not stat.S_ISDIR(path_state.st_mode):
        raise SessionError(f"{label} is not a directory: {path}")


def _require_regular_file(path: Path, label: str) -> None:
    path_state = _lstat_or_none(path)
    if path_state is None:
        raise SessionError(f"session is missing {label}: {path}")
    if stat.S_ISLNK(path_state.st_mode):
        raise SessionError(f"{label} is a symlink: {path}")
    if not stat.S_ISREG(path_state.st_mode):
        raise SessionError(f"{label} is not a regular file: {path}")


def _open_existing_regular_file(path: Path, flags: int, label: str) -> int:
    _require_regular_file(path, label)
    try:
        file_fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SessionError(f"could not open {label}: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise SessionError(f"{label} is not a regular file: {path}")
    except BaseException:
        os.close(file_fd)
        raise
    return file_fd


def _fsync_directory(directory: Path) -> None:
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _write_all(file_fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_fd, payload[offset:])
        if written <= 0:
            raise OSError("could not write complete file")
        offset += written


def _line_prefix(line_number: int | None) -> str:
    return "" if line_number is None else f"manifest line {line_number}: "


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value!r}")


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _copy_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise SessionError("metadata must be a dictionary")
    try:
        payload = _compact_json(metadata)
        payload.encode("utf-8")
        copied = json.loads(payload, parse_constant=_reject_json_constant)
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionError("metadata must be JSON-serializable UTF-8") from exc
    if not isinstance(copied, dict):
        raise SessionError("metadata must be a dictionary")
    return copied


def _copy_reason(reason: Any) -> str:
    if not isinstance(reason, str) or not reason:
        raise SessionError("rejection reason must be a nonempty string")
    try:
        payload = _compact_json(reason)
        payload.encode("utf-8")
        copied = json.loads(payload, parse_constant=_reject_json_constant)
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionError("rejection reason must be JSON-serializable UTF-8") from exc
    if not isinstance(copied, str) or not copied:
        raise SessionError("rejection reason must be a nonempty string")
    return copied
