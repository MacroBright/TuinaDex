"""Behavior tests for resumable stereo capture sessions."""

from __future__ import annotations

import json
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
