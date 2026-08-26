"""Durable, resumable storage for stereo calibration image pairs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_PAIR_FILE = re.compile(r"^pair_(\d+)_(left|right)\.png$")


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
        if session_dir.exists() or session_dir.is_symlink():
            raise SessionError(f"session already exists: {session_dir}")

        try:
            session_dir.mkdir(parents=True)
            (session_dir / "pairs").mkdir()
            (session_dir / "rejected").mkdir()
            (session_dir / "results").mkdir()
            (session_dir / "manifest.jsonl").touch(exist_ok=False)
        except OSError as exc:
            raise SessionError(f"could not create session {session_dir}") from exc
        return cls(session_dir)

    @classmethod
    def open(cls, session_dir: Path) -> "SessionStore":
        """Open and validate an existing capture session."""
        store = cls(Path(session_dir))
        store._require_layout()
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
        self._require_layout()
        events, _active = self._replay_manifest()
        pair_id = self._next_pair_id(events)
        left_path, right_path = self._pair_paths(self.pairs_dir, pair_id)
        left_tmp = _temporary_path(left_path)
        right_tmp = _temporary_path(right_path)
        targets = (left_path, right_path, left_tmp, right_tmp)
        if any(path.exists() or path.is_symlink() for path in targets):
            raise SessionError(f"refusing to overwrite files for pair {pair_id}")

        try:
            if not cv2.imwrite(str(left_tmp), left):
                raise SessionError(f"could not encode left image for pair {pair_id}")
            if not cv2.imwrite(str(right_tmp), right):
                raise SessionError(f"could not encode right image for pair {pair_id}")
            left_tmp.replace(left_path)
            right_tmp.replace(right_path)
            if not left_path.is_file() or not right_path.is_file():
                raise SessionError(f"could not publish both images for pair {pair_id}")
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
        self._require_layout()
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
        if not isinstance(reason, str) or not reason:
            raise SessionError("rejection reason must be a nonempty string")
        self._require_layout()
        _events, active = self._replay_manifest()
        if not active:
            raise SessionError("no active pair to reject")

        pair_id = max(active)
        metadata = active[pair_id]
        left_source, right_source = self._pair_paths(self.pairs_dir, pair_id)
        left_target, right_target = self._pair_paths(self.rejected_dir, pair_id)
        if any(path.exists() or path.is_symlink() for path in (left_target, right_target)):
            raise SessionError(f"refusing to overwrite rejected files for pair {pair_id}")

        try:
            left_source.replace(left_target)
            right_source.replace(right_target)
            if not left_target.is_file() or not right_target.is_file():
                raise SessionError(f"could not move both images for pair {pair_id}")
            self._append_event({"action": "reject", "pair_id": pair_id, "reason": reason})
        except SessionError:
            raise
        except OSError as exc:
            raise SessionError(f"could not reject pair {pair_id}") from exc

        return SavedPair(pair_id, left_target, right_target, _copy_metadata(metadata))

    def _require_layout(self) -> None:
        for directory in (self.session_dir, self.pairs_dir, self.rejected_dir, self.results_dir):
            if not directory.is_dir():
                raise SessionError(f"session is missing expected directory: {directory}")
        if not self.manifest_path.is_file():
            raise SessionError(f"session is missing manifest: {self.manifest_path}")

    def _replay_manifest(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        active: dict[int, dict[str, Any]] = {}
        active_lines: dict[int, int] = {}
        captured: set[int] = set()
        try:
            with self.manifest_path.open("r", encoding="utf-8") as manifest:
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
                    events.append(event)
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"could not read manifest {self.manifest_path}") from exc

        for pair_id in active:
            left_path, right_path = self._pair_paths(self.pairs_dir, pair_id)
            if left_path.exists() != right_path.exists():
                raise SessionError(
                    f"manifest line {active_lines[pair_id]}: active pair {pair_id} has "
                    "mismatched left/right image files (one is missing)"
                )
            if not left_path.is_file() or not right_path.is_file():
                raise SessionError(
                    f"manifest line {active_lines[pair_id]}: active pair {pair_id} is "
                    "missing image files"
                )
        return events, active

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
        if action == "capture" and not isinstance(event.get("metadata"), dict):
            raise SessionError(f"manifest line {line_number}: capture metadata must be an object")
        if action == "reject" and (
            not isinstance(event.get("reason"), str) or not event["reason"]
        ):
            raise SessionError(f"manifest line {line_number}: reject reason must be a nonempty string")
        return event

    def _next_pair_id(self, events: list[dict[str, Any]]) -> int:
        maximum = max((event["pair_id"] for event in events), default=0)
        for directory in (self.pairs_dir, self.rejected_dir):
            for candidate in directory.iterdir():
                match = _PAIR_FILE.fullmatch(candidate.name)
                if match and candidate.is_file():
                    maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    @staticmethod
    def _pair_paths(directory: Path, pair_id: int) -> tuple[Path, Path]:
        stem = f"pair_{pair_id:04d}"
        return directory / f"{stem}_left.png", directory / f"{stem}_right.png"

    def _append_event(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        try:
            with self.manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(payload + "\n")
                manifest.flush()
                os.fsync(manifest.fileno())
        except (OSError, UnicodeError) as exc:
            raise SessionError("could not append manifest event") from exc

    @staticmethod
    def _cleanup_temps(*paths: Path) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.stem}.tmp{final_path.suffix}")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value!r}")


def _copy_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise SessionError("metadata must be a dictionary")
    try:
        copied = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionError("metadata must be JSON-serializable") from exc
    if not isinstance(copied, dict):
        raise SessionError("metadata must be a dictionary")
    return copied
