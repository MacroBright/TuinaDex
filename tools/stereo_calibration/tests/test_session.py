"""Behavior tests for resumable stereo capture sessions."""

from __future__ import annotations

import errno
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from tools.stereo_calibration import session
from tools.stereo_calibration.session import SessionError, SessionStore


@pytest.fixture
def image() -> np.ndarray:
    return np.full((3, 4, 3), 127, dtype=np.uint8)


def test_create_makes_expected_layout_and_refuses_existing_session(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "run-01")

    assert store.session_dir == tmp_path / "run-01"
    assert store.pairs_dir.is_dir()
    assert store.rejected_dir.is_dir()
    assert store.results_dir.is_dir()
    assert store.manifest_path.read_text(encoding="utf-8") == ""

    with pytest.raises(SessionError, match="already exists"):
        SessionStore.create(tmp_path, "run-01")


@pytest.mark.parametrize("session_id", ["", ".", "..", "../escape", "nested/name", "a\\b"])
def test_create_rejects_unsafe_session_ids_without_escaping_root(
    tmp_path: Path, session_id: str
) -> None:
    with pytest.raises(SessionError, match="session ID"):
        SessionStore.create(tmp_path, session_id)

    assert list(tmp_path.iterdir()) == []


def test_first_save_uses_id_one_and_exact_file_names(tmp_path: Path, image: np.ndarray) -> None:
    store = SessionStore.create(tmp_path, "run")

    saved = store.save_pair(image, image, {"note": "first"})

    assert saved.pair_id == 1
    assert saved.left_path == store.pairs_dir / "pair_0001_left.png"
    assert saved.right_path == store.pairs_dir / "pair_0001_right.png"
    assert saved.left_path.is_file()
    assert saved.right_path.is_file()
    assert [json.loads(line)["action"] for line in store.manifest_path.read_text().splitlines()] == [
        "capture"
    ]


def test_reopen_resumes_with_next_id(tmp_path: Path, image: np.ndarray) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.save_pair(image, image, {})

    reopened = SessionStore.open(store.session_dir)
    saved = reopened.save_pair(image, image, {})

    assert saved.pair_id == 2
    assert saved.left_path.name == "pair_0002_left.png"


def test_concurrent_saves_allocate_unique_monotonic_ids_and_reopen(
    tmp_path: Path, image: np.ndarray
) -> None:
    store = SessionStore.create(tmp_path, "run")

    with ThreadPoolExecutor(max_workers=6) as executor:
        saved = list(executor.map(lambda _: store.save_pair(image, image, {}), range(12)))

    assert sorted(pair.pair_id for pair in saved) == list(range(1, 13))
    assert len(store.manifest_path.read_text(encoding="utf-8").splitlines()) == 12
    assert [pair.pair_id for pair in SessionStore.open(store.session_dir).active_pairs()] == list(
        range(1, 13)
    )


def test_interleaved_store_instances_refresh_the_next_id(tmp_path: Path, image: np.ndarray) -> None:
    first = SessionStore.create(tmp_path, "run")
    second = SessionStore.open(first.session_dir)

    assert first.save_pair(image, image, {}).pair_id == 1
    assert second.save_pair(image, image, {}).pair_id == 2


def test_concurrent_independent_store_instances_allocate_contiguous_ids(
    tmp_path: Path, image: np.ndarray
) -> None:
    first = SessionStore.create(tmp_path, "run")
    second = SessionStore.open(first.session_dir)
    stores = [first, second] * 6

    with ThreadPoolExecutor(max_workers=6) as executor:
        saved = list(executor.map(lambda store: store.save_pair(image, image, {}), stores))

    assert sorted(pair.pair_id for pair in saved) == list(range(1, 13))
    events = [json.loads(line) for line in first.manifest_path.read_text().splitlines()]
    assert [event["action"] for event in events] == ["capture"] * 12
    assert [pair.pair_id for pair in SessionStore.open(first.session_dir).active_pairs()] == list(
        range(1, 13)
    )


def test_reject_last_moves_pair_logs_event_and_removes_it_from_active(
    tmp_path: Path, image: np.ndarray
) -> None:
    store = SessionStore.create(tmp_path, "run")
    saved = store.save_pair(image, image, {"exposure": {"left": 4}})

    rejected = store.reject_last("blurred")

    assert rejected.pair_id == saved.pair_id
    assert rejected.left_path == store.rejected_dir / "pair_0001_left.png"
    assert rejected.right_path == store.rejected_dir / "pair_0001_right.png"
    assert rejected.left_path.is_file()
    assert rejected.right_path.is_file()
    assert not saved.left_path.exists()
    assert not saved.right_path.exists()
    assert store.active_pairs() == []
    events = [json.loads(line) for line in store.manifest_path.read_text().splitlines()]
    assert [event["action"] for event in events] == ["capture", "reject"]
    assert events[-1]["reason"] == "blurred"


def test_rejected_id_is_never_reused(tmp_path: Path, image: np.ndarray) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.save_pair(image, image, {})
    store.reject_last("bad")

    assert store.save_pair(image, image, {}).pair_id == 2


@pytest.mark.parametrize("side", ["left", "right"])
def test_open_refuses_missing_rejected_image(
    tmp_path: Path, image: np.ndarray, side: str
) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.save_pair(image, image, {})
    store.reject_last("bad")
    (store.rejected_dir / f"pair_0001_{side}.png").unlink()

    with pytest.raises(SessionError, match=r"pair 1.*missing rejected image"):
        SessionStore.open(store.session_dir)


def test_reject_last_refuses_when_there_is_no_active_pair(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "run")

    with pytest.raises(SessionError, match="no active"):
        store.reject_last("bad")


def test_open_reports_line_number_for_malformed_json(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.manifest_path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(SessionError, match=r"line 1"):
        SessionStore.open(store.session_dir)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([{"action": "capture", "pair_id": 1, "metadata": {}}] * 2, "duplicate capture"),
        ([{"action": "reject", "pair_id": 1, "reason": "bad"}], "no active"),
        ([{"action": "capture", "pair_id": 1, "metadata": {}}, {"action": "reject", "pair_id": 2, "reason": "bad"}], "no active"),
        ([{"action": "other", "pair_id": 1}], "action"),
    ],
)
def test_open_rejects_malformed_events_duplicate_capture_or_unknown_rejection(
    tmp_path: Path, events: list[dict[str, object]], expected: str
) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.manifest_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    with pytest.raises(SessionError, match=expected):
        SessionStore.open(store.session_dir)


def test_open_refuses_missing_one_active_image(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.manifest_path.write_text(
        json.dumps({"action": "capture", "pair_id": 1, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    (store.pairs_dir / "pair_0001_left.png").touch()

    with pytest.raises(SessionError, match=r"pair 1.*missing|missing.*pair 1"):
        SessionStore.open(store.session_dir)


@pytest.mark.parametrize("component", ["session", "pairs", "rejected", "results", "manifest", "lock"])
def test_open_refuses_symlinked_session_component(tmp_path: Path, component: str) -> None:
    store = SessionStore.create(tmp_path, "run")
    component_paths = {
        "pairs": store.pairs_dir,
        "rejected": store.rejected_dir,
        "results": store.results_dir,
        "manifest": store.manifest_path,
        "lock": store.session_dir / ".session.lock",
    }
    path = store.session_dir if component == "session" else component_paths[component]
    if component == "lock" and not path.exists():
        pytest.fail("session lock was not created")
    moved = tmp_path / f"{component}-backing"
    path.rename(moved)

    try:
        path.symlink_to(moved, target_is_directory=component in {"session", "pairs", "rejected", "results"})
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"platform cannot create symlinks: {exc}")

    with pytest.raises(SessionError, match="symlink"):
        SessionStore.open(store.session_dir)


@pytest.mark.parametrize("location", ["pairs", "rejected"])
def test_open_refuses_symlinked_expected_image_outside_session(
    tmp_path: Path, image: np.ndarray, location: str
) -> None:
    store = SessionStore.create(tmp_path, "run")
    saved = store.save_pair(image, image, {})
    if location == "rejected":
        saved = store.reject_last("bad")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    saved.left_path.unlink()
    try:
        saved.left_path.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"platform cannot create symlinks: {exc}")

    with pytest.raises(SessionError, match="symlink"):
        SessionStore.open(store.session_dir)


def test_save_refuses_to_overwrite_existing_final_or_temp_file(
    tmp_path: Path, image: np.ndarray
) -> None:
    store = SessionStore.create(tmp_path, "run")
    target = store.pairs_dir / "pair_0001_left.tmp.png"
    target.touch()

    with pytest.raises(SessionError, match="refusing to overwrite"):
        store.save_pair(image, image, {})

    assert target.exists()
    assert store.manifest_path.read_text() == ""


@pytest.mark.parametrize("side", ["left", "right"])
def test_save_refuses_final_file_created_after_candidate_id_selection(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    store = SessionStore.create(tmp_path, "run")
    candidate_id = store._next_pair_id([])
    target = store.pairs_dir / f"pair_{candidate_id:04d}_{side}.png"
    sentinel = b"existing image bytes"
    target.write_bytes(sentinel)
    monkeypatch.setattr(store, "_next_pair_id", lambda _events: candidate_id)

    with pytest.raises(SessionError, match="refusing to overwrite"):
        store.save_pair(image, image, {})

    assert target.read_bytes() == sentinel
    assert list(store.pairs_dir.iterdir()) == [target]
    assert store.manifest_path.read_text() == ""


def test_failed_left_encode_leaves_no_files_or_event(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    monkeypatch.setattr(session.cv2, "imwrite", lambda *_args: False)

    with pytest.raises(SessionError, match="encode left"):
        store.save_pair(image, image, {})

    assert list(store.pairs_dir.iterdir()) == []
    assert store.manifest_path.read_text() == ""


def test_failed_right_encode_cleans_left_temp_file(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    real_imwrite = session.cv2.imwrite
    calls = 0

    def fail_second_encode(path: str, frame: np.ndarray) -> bool:
        nonlocal calls
        calls += 1
        return real_imwrite(path, frame) if calls == 1 else False

    monkeypatch.setattr(session.cv2, "imwrite", fail_second_encode)

    with pytest.raises(SessionError, match="encode right"):
        store.save_pair(image, image, {})

    assert list(store.pairs_dir.iterdir()) == []
    assert store.manifest_path.read_text() == ""


def test_nonserializable_metadata_is_rejected_before_creating_files(
    tmp_path: Path, image: np.ndarray
) -> None:
    store = SessionStore.create(tmp_path, "run")

    with pytest.raises(SessionError, match="metadata"):
        store.save_pair(image, image, {"not_json": object()})

    assert list(store.pairs_dir.iterdir()) == []
    assert store.manifest_path.read_text() == ""


def test_unpaired_surrogate_metadata_is_rejected_before_image_io(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    calls = 0

    def record_image_write(*_args: object) -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(session.cv2, "imwrite", record_image_write)

    with pytest.raises(SessionError, match="metadata"):
        store.save_pair(image, image, {"bad": "\ud800"})

    assert calls == 0
    assert list(store.pairs_dir.iterdir()) == []
    assert store.manifest_path.read_text() == ""


def test_unpaired_surrogate_reject_reason_is_rejected_before_files_move(
    tmp_path: Path, image: np.ndarray
) -> None:
    store = SessionStore.create(tmp_path, "run")
    saved = store.save_pair(image, image, {})
    manifest_before = store.manifest_path.read_text(encoding="utf-8")

    with pytest.raises(SessionError, match="reason"):
        store.reject_last("\ud800")

    assert saved.left_path.is_file()
    assert saved.right_path.is_file()
    assert not (store.session_dir / ".reject-transaction.json").exists()
    assert store.manifest_path.read_text(encoding="utf-8") == manifest_before


@pytest.mark.parametrize(
    "metadata_json",
    [r'{"value":1e100000}', r'{"bad":"\ud800"}'],
)
def test_open_rejects_nonfinite_or_invalid_utf8_capture_metadata(
    tmp_path: Path, metadata_json: str
) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.manifest_path.write_text(
        f'{{"action":"capture","pair_id":1,"metadata":{metadata_json}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(SessionError, match=r"line 1.*pair 1.*metadata"):
        SessionStore.open(store.session_dir)


def test_open_recovers_partial_rejection_after_second_image_move_fails(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    saved = store.save_pair(image, image, {})
    original_replace = Path.replace

    def fail_right_move(path: Path, target: Path) -> Path:
        if path == saved.right_path and target == store.rejected_dir / "pair_0001_right.png":
            raise OSError("injected second move failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_right_move)

    with pytest.raises(SessionError, match="reject pair 1"):
        store.reject_last("bad")

    marker = store.session_dir / ".reject-transaction.json"
    assert marker.is_file()
    assert (store.rejected_dir / "pair_0001_left.png").is_file()
    assert saved.right_path.is_file()
    monkeypatch.setattr(Path, "replace", original_replace)

    reopened = SessionStore.open(store.session_dir)

    assert reopened.active_pairs() == []
    assert not marker.exists()
    assert [json.loads(line)["action"] for line in reopened.manifest_path.read_text().splitlines()] == [
        "capture",
        "reject",
    ]


def test_open_finishes_rejection_after_manifest_append_writes_then_fails(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.save_pair(image, image, {})
    original_append = store._append_event

    def append_then_fail(event: dict[str, object]) -> None:
        original_append(event)
        raise SessionError("injected manifest fsync failure")

    monkeypatch.setattr(store, "_append_event", append_then_fail)

    with pytest.raises(SessionError, match="manifest fsync"):
        store.reject_last("bad")

    marker = store.session_dir / ".reject-transaction.json"
    assert marker.is_file()
    assert [json.loads(line)["action"] for line in store.manifest_path.read_text().splitlines()] == [
        "capture",
        "reject",
    ]
    monkeypatch.setattr(store, "_append_event", original_append)

    reopened = SessionStore.open(store.session_dir)

    assert reopened.active_pairs() == []
    assert not marker.exists()
    assert [json.loads(line)["action"] for line in reopened.manifest_path.read_text().splitlines()] == [
        "capture",
        "reject",
    ]


def test_capture_interrupted_manifest_write_keeps_reopenable_old_manifest(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    real_write = session.os.write
    failed = False

    def partial_then_fail(file_descriptor: int, payload: bytes) -> int:
        nonlocal failed
        if b'"action":"capture"' in payload:
            failed = True
            return real_write(file_descriptor, payload[:5])
        if failed:
            raise OSError(errno.EIO, "injected manifest write failure")
        return real_write(file_descriptor, payload)

    monkeypatch.setattr(session.os, "write", partial_then_fail)

    with pytest.raises(SessionError, match="manifest"):
        store.save_pair(image, image, {})

    monkeypatch.setattr(session.os, "write", real_write)
    reopened = SessionStore.open(store.session_dir)

    assert reopened.active_pairs() == []
    assert reopened.manifest_path.read_bytes() == b""
    assert not (store.session_dir / ".manifest.jsonl.tmp").exists()
    assert reopened.save_pair(image, image, {}).pair_id == 2


def test_reject_interrupted_manifest_write_recovers_one_reject_event(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    store.save_pair(image, image, {})
    real_write = session.os.write
    failed = False

    def partial_then_fail(file_descriptor: int, payload: bytes) -> int:
        nonlocal failed
        if b'"action":"reject"' in payload:
            failed = True
            return real_write(file_descriptor, payload[:5])
        if failed:
            raise OSError(errno.EIO, "injected manifest write failure")
        return real_write(file_descriptor, payload)

    monkeypatch.setattr(session.os, "write", partial_then_fail)

    with pytest.raises(SessionError, match="manifest"):
        store.reject_last("bad")

    monkeypatch.setattr(session.os, "write", real_write)
    reopened = SessionStore.open(store.session_dir)

    assert reopened.active_pairs() == []
    assert not (store.session_dir / ".reject-transaction.json").exists()
    assert [json.loads(line)["action"] for line in reopened.manifest_path.read_text().splitlines()] == [
        "capture",
        "reject",
    ]


@pytest.mark.parametrize("failure", ["write", "fsync", "replace"])
def test_failed_marker_publication_cleans_temp_and_allows_retry(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store = SessionStore.create(tmp_path, "run")
    saved = store.save_pair(image, image, {})
    marker = store.session_dir / ".reject-transaction.json"
    marker_temp = store.session_dir / ".reject-transaction.tmp"

    with monkeypatch.context() as fault:
        if failure == "write":
            real_write = session.os.write

            def fail_marker_write(file_descriptor: int, payload: bytes) -> int:
                if b'"pair_id":1' in payload:
                    raise OSError(errno.EIO, "injected marker write failure")
                return real_write(file_descriptor, payload)

            fault.setattr(session.os, "write", fail_marker_write)
        elif failure == "fsync":
            real_fsync = session.os.fsync

            def fail_marker_fsync(file_descriptor: int) -> None:
                raise OSError(errno.EIO, "injected marker fsync failure")

            fault.setattr(session.os, "fsync", fail_marker_fsync)
        else:
            real_replace = session.os.replace

            def fail_marker_replace(source: Path, target: Path) -> None:
                if target == marker:
                    raise OSError(errno.EIO, "injected marker replace failure")
                real_replace(source, target)

            fault.setattr(session.os, "replace", fail_marker_replace)

        with pytest.raises(SessionError, match="marker"):
            store.reject_last("bad")

    assert saved.left_path.is_file()
    assert saved.right_path.is_file()
    assert not marker.exists()
    assert not marker_temp.exists()
    assert [json.loads(line)["action"] for line in store.manifest_path.read_text().splitlines()] == [
        "capture"
    ]

    assert SessionStore.open(store.session_dir).reject_last("bad").pair_id == 1


def test_open_cleans_regular_stale_marker_temp_without_published_marker(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "run")
    marker_temp = store.session_dir / ".reject-transaction.tmp"
    marker_temp.write_text('{"pair_id":1,"reason":"bad"}', encoding="utf-8")

    reopened = SessionStore.open(store.session_dir)

    assert reopened.session_dir == store.session_dir
    assert not marker_temp.exists()


def test_directory_fsync_eio_raises_session_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")

    def fail_fsync(_file_descriptor: int) -> None:
        raise OSError(errno.EIO, "injected directory fsync failure")

    monkeypatch.setattr(session.os, "fsync", fail_fsync)

    with pytest.raises(SessionError, match="fsync directory"):
        session._fsync_directory(store.pairs_dir)


def test_capture_fsyncs_pairs_directory_before_committing_manifest(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    original_directory_fsync = session._fsync_directory
    original_append = store._append_event
    pairs_synced = False

    def record_directory_fsync(directory: Path) -> None:
        nonlocal pairs_synced
        if directory == store.pairs_dir:
            pairs_synced = True
        original_directory_fsync(directory)

    def require_prior_image_sync(event: dict[str, object]) -> None:
        assert pairs_synced
        original_append(event)

    monkeypatch.setattr(session, "_fsync_directory", record_directory_fsync)
    monkeypatch.setattr(store, "_append_event", require_prior_image_sync)

    store.save_pair(image, image, {})


def test_capture_pairs_directory_fsync_failure_preserves_orphan_and_skips_id(
    tmp_path: Path, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore.create(tmp_path, "run")
    real_fsync = session.os.fsync

    def fail_only_directory_fsync(file_descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError(errno.EIO, "injected pairs directory fsync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(session.os, "fsync", fail_only_directory_fsync)

    with pytest.raises(SessionError, match="fsync directory"):
        store.save_pair(image, image, {})

    monkeypatch.setattr(session.os, "fsync", real_fsync)
    reopened = SessionStore.open(store.session_dir)
    assert reopened.active_pairs() == []
    assert reopened.save_pair(image, image, {}).pair_id == 2


def test_metadata_is_defensively_copied_on_save_and_read(tmp_path: Path, image: np.ndarray) -> None:
    store = SessionStore.create(tmp_path, "run")
    metadata = {"nested": {"value": 1}}
    saved = store.save_pair(image, image, metadata)
    metadata["nested"]["value"] = 2

    assert saved.metadata == {"nested": {"value": 1}}
    active = store.active_pairs()
    active[0].metadata["nested"]["value"] = 3
    assert store.active_pairs()[0].metadata == {"nested": {"value": 1}}


def test_crash_orphan_file_causes_next_id_to_skip_it(tmp_path: Path, image: np.ndarray) -> None:
    store = SessionStore.create(tmp_path, "run")
    (store.pairs_dir / "pair_0007_left.png").touch()

    saved = store.save_pair(image, image, {})

    assert saved.pair_id == 8
