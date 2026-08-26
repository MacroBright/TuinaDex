# Guided Stereo Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-guided, resumable stereo-checkerboard capture and calibration tool that runs on the Ubuntu laptop and produces validated OpenCV rectification artifacts for the later point-cloud phase.

**Architecture:** A single Ubuntu process owns both V4L2 cameras, normalizes logical camera 2 with a 180-degree rotation, detects the checkerboard, stores canonical image pairs, and serves a loopback-only browser interface through Python's standard HTTP server. Focused modules isolate configuration, detection, storage, camera acquisition, calibration, service state, and HTTP handling so all non-hardware behavior can be tested with synthetic data and fake cameras.

**Tech Stack:** Python 3.10, NumPy, OpenCV, Python standard-library `dataclasses`, `json`, `http.server`, HTML/CSS/JavaScript, pytest, Git, Ubuntu 22.04 V4L2.

**Design source:** `docs/superpowers/specs/2026-08-26-stereo-calibration-design.md`

---

## Planned File Structure

```text
TuinaDex/
├── .gitignore                                      # Ignore local calibration sessions
├── pyproject.toml                                  # Collect tool tests with pytest
└── tools/
    ├── __init__.py                                 # Make repository tools importable in tests
    └── stereo_calibration/
        ├── __init__.py                             # Public package metadata
        ├── README.md                               # Beginner operation guide
        ├── example_config.json                     # Approved physical-camera configuration
        ├── config.py                               # Typed configuration and validation
        ├── detection.py                            # Rotation, checkerboard, and image-quality checks
        ├── session.py                              # Atomic, resumable pair storage
        ├── calibration.py                          # OpenCV calibration, metrics, and artifacts
        ├── cameras.py                              # Dual-V4L2 ownership and acquisition
        ├── service.py                              # Thread-safe application state and actions
        ├── web.py                                  # Loopback HTTP API and static file delivery
        ├── main.py                                 # CLI entry point and process lifecycle
        ├── static/
        │   └── index.html                          # Live two-camera capture page
        └── tests/
            ├── __init__.py
            ├── helpers.py                          # Synthetic checkerboard observations and fakes
            ├── test_config.py
            ├── test_detection.py
            ├── test_session.py
            ├── test_calibration.py
            ├── test_cameras.py
            ├── test_main.py
            └── test_web.py
```

The implementation intentionally stays outside the LeRobot and dexterous-hand packages. Runtime image data lives under `data/stereo_calibration/` and remains untracked.

---

### Task 1: Package Skeleton and Validated Configuration

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `tools/__init__.py`
- Create: `tools/stereo_calibration/__init__.py`
- Create: `tools/stereo_calibration/config.py`
- Create: `tools/stereo_calibration/example_config.json`
- Create: `tools/stereo_calibration/tests/__init__.py`
- Create: `tools/stereo_calibration/tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Create `tools/stereo_calibration/tests/test_config.py` with tests that lock down the approved logical-camera mapping and reject unsafe ambiguity:

```python
import json
from pathlib import Path

import pytest

from tools.stereo_calibration.config import AppConfig, ConfigError


def valid_payload(tmp_path: Path) -> dict:
    return {
        "left": {
            "name": "usb_port_1",
            "device": "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0",
            "rotation_degrees": 0,
        },
        "right": {
            "name": "usb_port_2",
            "device": "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0",
            "rotation_degrees": 180,
        },
        "capture": {
            "width": 1280,
            "height": 960,
            "fps": 30,
            "fourcc": "MJPG",
            "v4l2_controls": {
                "auto_exposure": 1,
                "exposure_time_absolute": 100,
                "gain": 32,
                "white_balance_automatic": 0,
                "white_balance_temperature": 4600,
            },
        },
        "checkerboard": {"columns": 9, "rows": 6, "square_size_mm": 35.0},
        "quality": {
            "edge_margin_px": 12,
            "min_laplacian_variance": 60.0,
            "max_saturated_fraction": 0.45,
        },
        "data_root": str(tmp_path / "data"),
        "web": {"host": "127.0.0.1", "port": 8765},
        "minimum_pairs": 18,
        "target_pairs": 30,
        "baseline_reference_mm": 200.0,
    }


def test_loads_approved_camera_rotation(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_payload(tmp_path)), encoding="utf-8")
    config = AppConfig.from_json(path)
    assert config.right.rotation_degrees == 180
    assert config.checkerboard.corner_count == 54
    assert config.capture.image_size == (1280, 960)


def test_rejects_transient_video_number(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload["left"]["device"] = "/dev/video4"
    with pytest.raises(ConfigError, match="/dev/v4l/by-path"):
        AppConfig.from_mapping(payload)


def test_rejects_duplicate_camera_paths(tmp_path: Path) -> None:
    payload = valid_payload(tmp_path)
    payload["right"]["device"] = payload["left"]["device"]
    with pytest.raises(ConfigError, match="different devices"):
        AppConfig.from_mapping(payload)


def test_effective_mapping_round_trips_expanded_data_root(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_payload(tmp_path))
    restored = AppConfig.from_mapping(config.to_mapping())
    assert restored == config
    assert restored.to_mapping()["data_root"] == str(config.data_root)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m pytest tools/stereo_calibration/tests/test_config.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tools.stereo_calibration.config'`.

- [ ] **Step 3: Implement immutable configuration objects**

Create `config.py` with these public types and validation rules:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CameraConfig:
    name: str
    device: str
    rotation_degrees: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraConfig":
        camera = cls(
            name=str(data["name"]),
            device=str(data["device"]),
            rotation_degrees=int(data["rotation_degrees"]),
        )
        if not camera.device.startswith("/dev/v4l/by-path/"):
            raise ConfigError("camera device must use /dev/v4l/by-path")
        if camera.rotation_degrees not in (0, 180):
            raise ConfigError("camera rotation must be 0 or 180 degrees")
        return camera


@dataclass(frozen=True)
class CaptureConfig:
    width: int
    height: int
    fps: int
    fourcc: str
    v4l2_controls: Mapping[str, int]

    @property
    def image_size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class CheckerboardConfig:
    columns: int
    rows: int
    square_size_mm: float

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.columns, self.rows

    @property
    def corner_count(self) -> int:
        return self.columns * self.rows


@dataclass(frozen=True)
class QualityConfig:
    edge_margin_px: int
    min_laplacian_variance: float
    max_saturated_fraction: float


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int


@dataclass(frozen=True)
class AppConfig:
    left: CameraConfig
    right: CameraConfig
    capture: CaptureConfig
    checkerboard: CheckerboardConfig
    quality: QualityConfig
    data_root: Path
    web: WebConfig
    minimum_pairs: int
    target_pairs: int
    baseline_reference_mm: float

    @classmethod
    def from_json(cls, path: Path) -> "AppConfig":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        left = CameraConfig.from_mapping(data["left"])
        right = CameraConfig.from_mapping(data["right"])
        if left.device == right.device:
            raise ConfigError("left and right cameras must use different devices")
        capture = CaptureConfig(**data["capture"])
        checkerboard = CheckerboardConfig(**data["checkerboard"])
        quality = QualityConfig(**data["quality"])
        web = WebConfig(**data["web"])
        config = cls(
            left=left,
            right=right,
            capture=capture,
            checkerboard=checkerboard,
            quality=quality,
            data_root=Path(data["data_root"]).expanduser(),
            web=web,
            minimum_pairs=int(data["minimum_pairs"]),
            target_pairs=int(data["target_pairs"]),
            baseline_reference_mm=float(data["baseline_reference_mm"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.capture.width <= 0 or self.capture.height <= 0 or self.capture.fps <= 0:
            raise ConfigError("capture width, height, and fps must be positive")
        if len(self.capture.fourcc) != 4:
            raise ConfigError("capture fourcc must contain four characters")
        if not self.capture.v4l2_controls or not all(
            isinstance(name, str) and isinstance(value, int)
            for name, value in self.capture.v4l2_controls.items()
        ):
            raise ConfigError("v4l2 controls must be a non-empty string-to-integer mapping")
        if self.checkerboard.columns < 2 or self.checkerboard.rows < 2:
            raise ConfigError("checkerboard dimensions must be at least 2x2")
        if self.checkerboard.square_size_mm <= 0:
            raise ConfigError("checkerboard square size must be positive")
        if not 0.0 < self.quality.max_saturated_fraction < 1.0:
            raise ConfigError("max saturated fraction must be between 0 and 1")
        if self.minimum_pairs < 3 or self.target_pairs < self.minimum_pairs:
            raise ConfigError("target pairs must be greater than or equal to minimum pairs")
        if self.web.host != "127.0.0.1":
            raise ConfigError("web server must bind to 127.0.0.1")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "left": {
                "name": self.left.name,
                "device": self.left.device,
                "rotation_degrees": self.left.rotation_degrees,
            },
            "right": {
                "name": self.right.name,
                "device": self.right.device,
                "rotation_degrees": self.right.rotation_degrees,
            },
            "capture": {
                "width": self.capture.width,
                "height": self.capture.height,
                "fps": self.capture.fps,
                "fourcc": self.capture.fourcc,
                "v4l2_controls": dict(self.capture.v4l2_controls),
            },
            "checkerboard": {
                "columns": self.checkerboard.columns,
                "rows": self.checkerboard.rows,
                "square_size_mm": self.checkerboard.square_size_mm,
            },
            "quality": {
                "edge_margin_px": self.quality.edge_margin_px,
                "min_laplacian_variance": self.quality.min_laplacian_variance,
                "max_saturated_fraction": self.quality.max_saturated_fraction,
            },
            "data_root": str(self.data_root),
            "web": {"host": self.web.host, "port": self.web.port},
            "minimum_pairs": self.minimum_pairs,
            "target_pairs": self.target_pairs,
            "baseline_reference_mm": self.baseline_reference_mm,
        }
```

Create the package `__init__.py` files and set `__version__ = "0.1.0"` in `tools/stereo_calibration/__init__.py`.

- [ ] **Step 4: Add the approved example configuration and ignore runtime data**

Create `example_config.json` using the exact values from `valid_payload`, with:

```json
{
  "left": {
    "name": "usb_port_1",
    "device": "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0",
    "rotation_degrees": 0
  },
  "right": {
    "name": "usb_port_2",
    "device": "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0",
    "rotation_degrees": 180
  },
  "capture": {
    "width": 1280,
    "height": 960,
    "fps": 30,
    "fourcc": "MJPG",
    "v4l2_controls": {
      "auto_exposure": 1,
      "exposure_time_absolute": 100,
      "gain": 32,
      "white_balance_automatic": 0,
      "white_balance_temperature": 4600
    }
  },
  "checkerboard": {"columns": 9, "rows": 6, "square_size_mm": 35.0},
  "quality": {
    "edge_margin_px": 12,
    "min_laplacian_variance": 60.0,
    "max_saturated_fraction": 0.45
  },
  "data_root": "~/projects/TuinaDex/data/stereo_calibration",
  "web": {"host": "127.0.0.1", "port": 8765},
  "minimum_pairs": 18,
  "target_pairs": 30,
  "baseline_reference_mm": 200.0
}
```

Append `data/stereo_calibration/` and `.superpowers/` to `.gitignore`. Add `"tools"` to `[tool.pytest.ini_options].testpaths` in `pyproject.toml`.

- [ ] **Step 5: Run configuration tests and repository checks**

Run:

```bash
python -m pytest tools/stereo_calibration/tests/test_config.py -v
python -m pytest --collect-only -q
git diff --check
```

Expected: configuration tests pass; tool tests are collected; `git diff --check` is silent.

- [ ] **Step 6: Commit the configuration foundation**

```bash
git add .gitignore pyproject.toml tools/__init__.py tools/stereo_calibration
git commit -m "feat: add stereo calibration configuration"
```

---

### Task 2: Frame Normalization and Checkerboard Quality Detection

**Files:**
- Create: `tools/stereo_calibration/detection.py`
- Create: `tools/stereo_calibration/tests/helpers.py`
- Create: `tools/stereo_calibration/tests/test_detection.py`

- [ ] **Step 1: Write failing rotation, object-point, and detection tests**

Create tests for the three public functions `normalize_frame`, `make_object_points`, and `detect_checkerboard`:

```python
import cv2
import numpy as np

from tools.stereo_calibration.config import CheckerboardConfig, QualityConfig
from tools.stereo_calibration.detection import (
    detect_checkerboard,
    make_object_points,
    normalize_frame,
)


BOARD = CheckerboardConfig(columns=9, rows=6, square_size_mm=35.0)
QUALITY = QualityConfig(
    edge_margin_px=12,
    min_laplacian_variance=10.0,
    max_saturated_fraction=0.60,
)


def synthetic_checkerboard(square_px: int = 60) -> np.ndarray:
    board = np.zeros(((BOARD.rows + 1) * square_px, (BOARD.columns + 1) * square_px), np.uint8)
    for row in range(BOARD.rows + 1):
        for col in range(BOARD.columns + 1):
            if (row + col) % 2 == 0:
                y0, y1 = row * square_px, (row + 1) * square_px
                x0, x1 = col * square_px, (col + 1) * square_px
                board[y0:y1, x0:x1] = 255
    canvas = np.full((960, 1280), 127, np.uint8)
    y0, x0 = 220, 340
    canvas[y0 : y0 + board.shape[0], x0 : x0 + board.shape[1]] = board
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_normalize_frame_rotates_right_camera_180_degrees() -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    normalized = normalize_frame(frame, 180)
    assert np.array_equal(normalized, cv2.rotate(frame, cv2.ROTATE_180))


def test_object_points_use_verified_millimetre_scale() -> None:
    points = make_object_points(BOARD)
    assert points.shape == (54, 3)
    assert tuple(points[1]) == (35.0, 0.0, 0.0)
    assert tuple(points[9]) == (0.0, 35.0, 0.0)


def test_detects_all_54_inner_corners() -> None:
    result = detect_checkerboard(synthetic_checkerboard(), BOARD, QUALITY)
    assert result.found
    assert result.corner_count == 54
    assert result.saveable
    assert result.reasons == ()
```

- [ ] **Step 2: Run the tests and verify missing-function failures**

```bash
python -m pytest tools/stereo_calibration/tests/test_detection.py -v
```

Expected: import fails because `detection.py` does not exist.

- [ ] **Step 3: Implement normalized frames and detection results**

Create `detection.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import CheckerboardConfig, QualityConfig


@dataclass(frozen=True)
class DetectionResult:
    found: bool
    corners: NDArray[np.float32] | None
    sharpness: float
    saturated_fraction: float
    border_ok: bool
    saveable: bool
    reasons: tuple[str, ...]

    @property
    def corner_count(self) -> int:
        return 0 if self.corners is None else int(len(self.corners))


def normalize_frame(frame: NDArray[np.uint8], rotation_degrees: int) -> NDArray[np.uint8]:
    if rotation_degrees == 0:
        return frame
    if rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"unsupported rotation: {rotation_degrees}")


def make_object_points(board: CheckerboardConfig) -> NDArray[np.float32]:
    grid = np.zeros((board.corner_count, 3), np.float32)
    grid[:, :2] = np.mgrid[0 : board.columns, 0 : board.rows].T.reshape(-1, 2)
    grid[:, :2] *= board.square_size_mm
    return grid


def detect_checkerboard(
    frame: NDArray[np.uint8],
    board: CheckerboardConfig,
    quality: QualityConfig,
) -> DetectionResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    found, corners = cv2.findChessboardCornersSB(gray, board.pattern_size, flags)
    saturated_fraction = float(np.mean(gray >= 250))
    reasons: list[str] = []
    border_ok = False
    sharpness = 0.0
    if not found or corners is None or len(corners) != board.corner_count:
        reasons.append(f"expected {board.corner_count} corners")
        corners = None
    else:
        xy = corners.reshape(-1, 2)
        x0, y0 = np.floor(xy.min(axis=0)).astype(int)
        x1, y1 = np.ceil(xy.max(axis=0)).astype(int)
        h, w = gray.shape
        border_ok = (
            x0 >= quality.edge_margin_px
            and y0 >= quality.edge_margin_px
            and x1 < w - quality.edge_margin_px
            and y1 < h - quality.edge_margin_px
        )
        if not border_ok:
            reasons.append("checkerboard is too close to an image edge")
        roi = gray[max(y0, 0) : min(y1 + 1, h), max(x0, 0) : min(x1 + 1, w)]
        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
        if sharpness < quality.min_laplacian_variance:
            reasons.append("checkerboard is too blurry")
    if saturated_fraction > quality.max_saturated_fraction:
        reasons.append("image is severely overexposed")
    return DetectionResult(
        found=corners is not None,
        corners=corners,
        sharpness=sharpness,
        saturated_fraction=saturated_fraction,
        border_ok=border_ok,
        saveable=corners is not None and not reasons,
        reasons=tuple(reasons),
    )
```

- [ ] **Step 4: Add rejection tests**

Add tests proving that a blurred board and a board against the image edge are detected but are not saveable. Assert the exact reason strings so browser diagnostics remain stable.

- [ ] **Step 5: Run focused and full tool tests**

```bash
python -m pytest tools/stereo_calibration/tests/test_detection.py -v
python -m pytest tools/stereo_calibration/tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit detection behavior**

```bash
git add tools/stereo_calibration/detection.py tools/stereo_calibration/tests
git commit -m "feat: detect stereo calibration checkerboards"
```

---

### Task 3: Atomic and Resumable Calibration Sessions

**Files:**
- Create: `tools/stereo_calibration/session.py`
- Create: `tools/stereo_calibration/tests/test_session.py`

- [ ] **Step 1: Write failing session-storage tests**

The tests must verify pair numbering, canonical image names, restart recovery, and explicit non-destructive rejection:

```python
import json
from pathlib import Path

import numpy as np

from tools.stereo_calibration.session import SessionStore


def frame(value: int) -> np.ndarray:
    return np.full((20, 30, 3), value, np.uint8)


def metadata() -> dict:
    return {"left_timestamp_ns": 100, "right_timestamp_ns": 105, "corners": 54}


def test_resume_uses_next_unused_pair_number(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "20260826-test")
    first = store.save_pair(frame(10), frame(20), metadata())
    reopened = SessionStore.open(tmp_path / "20260826-test")
    second = reopened.save_pair(frame(30), frame(40), metadata())
    assert first.pair_id == 1
    assert second.pair_id == 2
    assert second.left_path.name == "pair_0002_left.png"


def test_reject_last_moves_files_and_records_event(tmp_path: Path) -> None:
    store = SessionStore.create(tmp_path, "20260826-test")
    saved = store.save_pair(frame(10), frame(20), metadata())
    rejected = store.reject_last("operator requested deletion")
    assert rejected.pair_id == saved.pair_id
    assert not saved.left_path.exists()
    assert (store.rejected_dir / saved.left_path.name).exists()
    events = [json.loads(line) for line in store.manifest_path.read_text().splitlines()]
    assert [event["action"] for event in events] == ["capture", "reject"]
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
python -m pytest tools/stereo_calibration/tests/test_session.py -v
```

Expected: import fails because `SessionStore` does not exist.

- [ ] **Step 3: Implement `SessionStore` with atomic writes**

Create:

```python
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SavedPair:
    pair_id: int
    left_path: Path
    right_path: Path
    metadata: dict[str, Any]


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SessionError(f"malformed manifest line {line_number}") from exc
        if event.get("action") not in ("capture", "reject") or not isinstance(
            event.get("pair_id"), int
        ):
            raise SessionError(f"invalid manifest event on line {line_number}")
        events.append(event)
    return events


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_png_atomically(path: Path, image: NDArray[np.uint8]) -> None:
    temporary = path.with_name(path.stem + ".tmp.png")
    if temporary.exists() or path.exists():
        raise SessionError(f"refusing to overwrite image: {path.name}")
    if not cv2.imwrite(str(temporary), image):
        raise SessionError(f"failed to encode image: {path.name}")
    temporary.replace(path)


def _replay_active_pairs(
    events: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    active: dict[int, dict[str, Any]] = {}
    for event in events:
        pair_id = int(event["pair_id"])
        if event["action"] == "capture":
            if pair_id in active:
                raise SessionError(f"duplicate capture event for pair {pair_id}")
            active[pair_id] = dict(event.get("metadata", {}))
        else:
            if pair_id not in active:
                raise SessionError(f"reject event has no active pair {pair_id}")
            del active[pair_id]
    return active


class SessionStore:
    def __init__(self, session_dir: Path, events: list[dict[str, Any]]) -> None:
        self.session_dir = session_dir
        self.pairs_dir = session_dir / "pairs"
        self.rejected_dir = session_dir / "rejected"
        self.results_dir = session_dir / "results"
        self.manifest_path = session_dir / "manifest.jsonl"
        self._events = events
        self._next_pair_id = max((int(event["pair_id"]) for event in events), default=0) + 1

    @classmethod
    def create(cls, data_root: Path, session_id: str) -> "SessionStore":
        session_dir = data_root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        for name in ("pairs", "rejected", "results"):
            (session_dir / name).mkdir()
        (session_dir / "manifest.jsonl").touch()
        return cls.open(session_dir)

    @classmethod
    def open(cls, session_dir: Path) -> "SessionStore":
        manifest = session_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise SessionError(f"manifest not found: {manifest}")
        events = _read_events(manifest)
        return cls(session_dir=session_dir, events=events)

    def save_pair(
        self,
        left: NDArray[np.uint8],
        right: NDArray[np.uint8],
        metadata: dict[str, Any],
    ) -> SavedPair:
        pair_id = self._next_pair_id
        left_path = self.pairs_dir / f"pair_{pair_id:04d}_left.png"
        right_path = self.pairs_dir / f"pair_{pair_id:04d}_right.png"
        if left_path.exists() or right_path.exists():
            raise SessionError(f"pair {pair_id} already exists")
        _write_png_atomically(left_path, left)
        try:
            _write_png_atomically(right_path, right)
        except Exception:
            left_path.replace(self.rejected_dir / left_path.name)
            raise
        event = {"action": "capture", "pair_id": pair_id, "metadata": metadata}
        _append_event(self.manifest_path, event)
        self._events.append(event)
        self._next_pair_id += 1
        return SavedPair(pair_id, left_path, right_path, metadata)

    def active_pairs(self) -> list[SavedPair]:
        active = _replay_active_pairs(self._events)
        return [self._saved_pair(pair_id, metadata) for pair_id, metadata in active.items()]

    def _saved_pair(self, pair_id: int, metadata: dict[str, Any]) -> SavedPair:
        return SavedPair(
            pair_id=pair_id,
            left_path=self.pairs_dir / f"pair_{pair_id:04d}_left.png",
            right_path=self.pairs_dir / f"pair_{pair_id:04d}_right.png",
            metadata=metadata,
        )

    def reject_last(self, reason: str) -> SavedPair:
        active = self.active_pairs()
        if not active:
            raise SessionError("no active pairs to reject")
        pair = active[-1]
        pair.left_path.replace(self.rejected_dir / pair.left_path.name)
        pair.right_path.replace(self.rejected_dir / pair.right_path.name)
        event = {"action": "reject", "pair_id": pair.pair_id, "reason": reason}
        _append_event(self.manifest_path, event)
        self._events.append(event)
        return pair
```

Implementation rules:

- `create` uses `mkdir(parents=True, exist_ok=False)` and creates `pairs/`, `rejected/`, and `results/`.
- `open` reconstructs active pair state by replaying `manifest.jsonl`; it refuses malformed events rather than guessing.
- `save_pair` writes `*.tmp.png` with `cv2.imwrite`, checks the returned boolean, then uses `Path.replace` to publish both final images.
- Only after both images exist does it append and `fsync` a `capture` event.
- `reject_last` moves the two active files into `rejected/` and appends a `reject` event. It never recursively deletes.
- Next pair ID is one greater than the highest ID ever present in the event log, so rejected IDs are not reused.

- [ ] **Step 4: Add failure-recovery tests**

Add tests for a malformed manifest line, rejection with no active pairs, and refusal to overwrite an existing final image. Use `pytest.raises` with exact exception messages.

- [ ] **Step 5: Run session and full tests**

```bash
python -m pytest tools/stereo_calibration/tests/test_session.py -v
python -m pytest tools/stereo_calibration/tests -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit resumable storage**

```bash
git add tools/stereo_calibration/session.py tools/stereo_calibration/tests/test_session.py
git commit -m "feat: add resumable stereo capture sessions"
```

---

### Task 4: Calibration Mathematics, Metrics, and Artifacts

**Files:**
- Create: `tools/stereo_calibration/calibration.py`
- Modify: `tools/stereo_calibration/tests/helpers.py`
- Create: `tools/stereo_calibration/tests/test_calibration.py`

- [ ] **Step 1: Build deterministic synthetic stereo observations**

In `tests/helpers.py`, generate at least 20 board poses using fixed rotations and depths from 600 to 950 mm. Use:

```python
K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]])
distortion = np.zeros(5)
R_left_to_right = np.eye(3)
T_left_to_right = np.array([[-200.0], [0.0], [0.0]])
```

For each board pose, project the same millimetre object points into each camera with `cv2.projectPoints`. Compose the right pose as:

```python
R_right_board = R_left_to_right @ R_left_board
t_right_board = R_left_to_right @ t_left_board + T_left_to_right
```

Return object-point lists and matching `(N, 1, 2)` float32 image-point lists.

- [ ] **Step 2: Write failing calibration tests**

```python
import numpy as np

from tools.stereo_calibration.calibration import calibrate_observations
from tools.stereo_calibration.tests.helpers import synthetic_stereo_observations


def test_recovers_approximately_200_mm_baseline() -> None:
    observations = synthetic_stereo_observations()
    result = calibrate_observations(observations, image_size=(1280, 960))
    assert result.pair_count == 20
    assert abs(result.baseline_mm - 200.0) < 2.0
    assert result.stereo_rms < 0.1
    assert result.vertical_error_median_px < 0.1
    assert result.vertical_error_p95_px < 0.1


def test_rejects_fewer_than_minimum_pairs() -> None:
    observations = synthetic_stereo_observations()[:17]
    with pytest.raises(ValueError, match="at least 18 valid pairs"):
        calibrate_observations(observations, image_size=(1280, 960), minimum_pairs=18)
```

- [ ] **Step 3: Run tests and verify missing implementation failure**

```bash
python -m pytest tools/stereo_calibration/tests/test_calibration.py -v
```

Expected: import fails because `calibration.py` does not exist.

- [ ] **Step 4: Implement calibration data types and core computation**

Define:

```python
@dataclass(frozen=True)
class StereoObservation:
    pair_id: int
    object_points: NDArray[np.float32]
    left_corners: NDArray[np.float32]
    right_corners: NDArray[np.float32]


@dataclass(frozen=True)
class CalibrationResult:
    pair_count: int
    left_rms: float
    right_rms: float
    stereo_rms: float
    left_camera_matrix: NDArray[np.float64]
    right_camera_matrix: NDArray[np.float64]
    left_distortion: NDArray[np.float64]
    right_distortion: NDArray[np.float64]
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    essential: NDArray[np.float64]
    fundamental: NDArray[np.float64]
    rectification_left: NDArray[np.float64]
    rectification_right: NDArray[np.float64]
    projection_left: NDArray[np.float64]
    projection_right: NDArray[np.float64]
    disparity_to_depth: NDArray[np.float64]
    map_left_x: NDArray[np.float32]
    map_left_y: NDArray[np.float32]
    map_right_x: NDArray[np.float32]
    map_right_y: NDArray[np.float32]
    per_pair_errors: dict[int, dict[str, float]]
    vertical_error_median_px: float
    vertical_error_p95_px: float

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.translation))
```

`calibrate_observations` must:

1. enforce the minimum pair count and consistent shapes;
2. call `cv2.calibrateCamera` separately for both cameras;
3. compute per-pair monocular reprojection errors with `cv2.projectPoints`;
4. call `cv2.stereoCalibrate` with every observation list, both calibrated intrinsics, `image_size`, and `flags=cv2.CALIB_FIX_INTRINSIC`;
5. call `cv2.stereoRectify` with both intrinsics, `image_size`, the recovered `R` and `T`, `flags=cv2.CALIB_ZERO_DISPARITY`, and `alpha=0`;
6. create four fixed-size float remaps with `cv2.initUndistortRectifyMap` and map type `cv2.CV_32FC1`;
7. rectify corresponding corners with `cv2.undistortPoints` and calculate absolute vertical differences;
8. report median and 95th-percentile vertical differences;
9. return all arrays without writing files.

Also implement:

```python
def load_session_observations(
    session: SessionStore,
    board: CheckerboardConfig,
) -> tuple[list[StereoObservation], list[dict[str, str | int]]]:
    """Return valid observations plus explicit skip diagnostics."""
```

For every active `SavedPair`, require both image files, both metadata corner arrays with shape `(54, 2)`, and identical configured image dimensions. Convert corners to `(54, 1, 2)` float32 arrays and pair them with a fresh copy of `make_object_points(board)`. Add malformed IDs to the diagnostics list with an exact reason and do not guess or pair files by directory order. The service passes only returned observations to `calibrate_observations`, while reports include the skip diagnostics.

- [ ] **Step 5: Implement artifact writing without overwrite**

Add:

```python
def write_calibration_run(
    session: SessionStore,
    result: CalibrationResult,
    config: AppConfig,
    source_pairs: list[SavedPair],
) -> Path:
    """Create one new results/YYYYMMDD-HHMMSSffffff directory and return it."""
```

The function writes:

- every numeric matrix/remap to `stereo_calibration.npz`;
- interoperable matrices and metadata to `stereo_calibration.yaml` using `cv2.FileStorage`;
- exact metrics, thresholds, pair IDs, and warning/failure status to `report.json`;
- the same results in plain Chinese to `report.md`;
- at least three rectified side-by-side previews with horizontal lines every 40 pixels.

Mark a pair as a suspected outlier when its combined left/right per-pair reprojection error is greater than `max(1.5, median_error + 3 * median_absolute_deviation)`. Store the pair ID and measured error in both reports, but keep it in the computation and never remove its images automatically.

Status logic must be deterministic:

```python
if max(left_rms, right_rms, stereo_rms) > 1.5:
    status = "fail"
elif max(left_rms, right_rms, stereo_rms) > 1.0:
    status = "warning"
elif vertical_error_median_px >= 1.0 or vertical_error_p95_px >= 2.0:
    status = "warning"
else:
    status = "pass"
```

Include a baseline warning when `abs(result.baseline_mm - config.baseline_reference_mm) > 0.15 * config.baseline_reference_mm`.

- [ ] **Step 6: Test artifact round trips and no-overwrite behavior**

Assert that required NPZ keys are loadable, YAML matrices have their expected shapes, report status is `pass` for clean synthetic data, and two calls produce distinct result directories. Add one malformed saved-pair metadata fixture and assert that `load_session_observations` returns its pair ID and reason in skip diagnostics rather than silently pairing or crashing.

- [ ] **Step 7: Run calibration and full tests**

```bash
python -m pytest tools/stereo_calibration/tests/test_calibration.py -v
python -m pytest tools/stereo_calibration/tests -q
```

Expected: all tests pass and synthetic baseline recovery is within 2 mm.

- [ ] **Step 8: Commit calibration engine**

```bash
git add tools/stereo_calibration/calibration.py tools/stereo_calibration/tests
git commit -m "feat: compute and validate stereo calibration"
```

---

### Task 5: Dual-Camera Ownership and Acquisition

**Files:**
- Create: `tools/stereo_calibration/cameras.py`
- Create: `tools/stereo_calibration/tests/test_cameras.py`

- [ ] **Step 1: Write fake-capture tests before hardware code**

Define a fake `cv2.VideoCapture` implementation in the test. Lock down call order (`grab` both before either `retrieve`), right-camera normalization, and mode validation:

```python
def test_reads_complete_pair_and_rotates_right_frame_180() -> None:
    log: list[str] = []
    left_frame = coordinate_frame()
    right_frame = coordinate_frame(offset=50)
    pair = OpenCVCameraPair(
        config,
        capture_factory=fake_factory(log, left_frame, right_frame),
        control_runner=lambda device, controls: None,
    )
    pair.open()
    captured = pair.read()
    assert log.index("left.grab") < log.index("left.retrieve")
    assert log.index("right.grab") < log.index("left.retrieve")
    assert np.array_equal(captured.right, cv2.rotate(right_frame, cv2.ROTATE_180))


def test_refuses_unexpected_resolution() -> None:
    pair = OpenCVCameraPair(
        config,
        capture_factory=fake_factory_with_size(640, 480),
        control_runner=lambda device, controls: None,
    )
    with pytest.raises(CameraError, match="expected 1280x960"):
        pair.open()


def test_applies_the_same_v4l2_controls_to_both_devices() -> None:
    calls: list[tuple[str, dict[str, int]]] = []
    pair = OpenCVCameraPair(
        config,
        capture_factory=fake_factory_with_size(1280, 960),
        control_runner=lambda device, controls: calls.append((device, dict(controls))),
    )
    pair.open()
    assert calls == [
        (config.left.device, dict(config.capture.v4l2_controls)),
        (config.right.device, dict(config.capture.v4l2_controls)),
    ]
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
python -m pytest tools/stereo_calibration/tests/test_cameras.py -v
```

Expected: import fails because `cameras.py` does not exist.

- [ ] **Step 3: Implement the camera-pair interface**

Create these public types:

```python
class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class FramePair:
    left: NDArray[np.uint8]
    right: NDArray[np.uint8]
    left_timestamp_ns: int
    right_timestamp_ns: int


class PairSource(Protocol):
    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> FramePair:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVCameraPair:
    def __init__(
        self,
        config: AppConfig,
        capture_factory: Callable[[str], cv2.VideoCapture] | None = None,
        control_runner: Callable[[str, Mapping[str, int]], None] = apply_v4l2_controls,
    ) -> None:
        self.config = config
        self.capture_factory = capture_factory
        self.control_runner = control_runner
        self.left_capture = None
        self.right_capture = None

    def _open_one(self, device: str) -> cv2.VideoCapture:
        self.control_runner(device, self.config.capture.v4l2_controls)
        if self.capture_factory is None:
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        else:
            capture = self.capture_factory(device)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"failed to open camera: {device}")
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.config.capture.fourcc),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.capture.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if (actual_width, actual_height) != self.config.capture.image_size:
            capture.release()
            raise CameraError(
                f"expected {self.config.capture.width}x{self.config.capture.height}, "
                f"got {actual_width}x{actual_height} for {device}"
            )
        return capture

    def open(self) -> None:
        self.left_capture = self._open_one(self.config.left.device)
        try:
            self.right_capture = self._open_one(self.config.right.device)
        except Exception:
            self.left_capture.release()
            self.left_capture = None
            raise

    def read(self) -> FramePair:
        if self.left_capture is None or self.right_capture is None:
            raise CameraError("camera pair is not open")
        if not self.left_capture.grab() or not self.right_capture.grab():
            raise CameraError("failed to grab a complete stereo pair")
        left_time = time.monotonic_ns()
        left_ok, left = self.left_capture.retrieve()
        right_time = time.monotonic_ns()
        right_ok, right = self.right_capture.retrieve()
        if not left_ok or not right_ok:
            raise CameraError("failed to retrieve a complete stereo pair")
        return FramePair(
            left=normalize_frame(left, self.config.left.rotation_degrees),
            right=normalize_frame(right, self.config.right.rotation_degrees),
            left_timestamp_ns=left_time,
            right_timestamp_ns=right_time,
        )

    def close(self) -> None:
        for capture in (self.left_capture, self.right_capture):
            if capture is not None:
                capture.release()
        self.left_capture = None
        self.right_capture = None
```

The complete file imports `subprocess`, `time`, `Mapping`, `Protocol`, `Callable`, OpenCV, NumPy types, configuration types, and `normalize_frame`. Implement `apply_v4l2_controls(device, controls)` by ordering mode controls first (`auto_exposure`, then `white_balance_automatic`) followed by the remaining sorted keys, joining them as `name=value` pairs, and running this argv list without a shell:

```python
subprocess.run(
    ["v4l2-ctl", f"--device={device}", f"--set-ctrl={control_text}"],
    check=True,
    capture_output=True,
    text=True,
)
```

Immediately run a second shell-free argv command, `["v4l2-ctl", f"--device={device}", f"--get-ctrl={control_names}"]`, parse every returned integer, and raise `CameraError` if any applied value differs from configuration. Wrap `FileNotFoundError` and `subprocess.CalledProcessError` as `CameraError` containing the device path and stderr. The same `capture.v4l2_controls` mapping is applied independently to both devices before opening them. Production captures use `cv2.VideoCapture(device, cv2.CAP_V4L2)`. Set FOURCC, width, height, and FPS; read back FOURCC, width, height, and FPS; reject a resolution or FOURCC mismatch and reject an FPS difference greater than 1.0. On `read`, grab left and right before retrieving either. Timestamp each retrieval with `time.monotonic_ns`. Normalize camera 2 immediately and raise `CameraError` if either operation fails.

- [ ] **Step 4: Implement a single-owner worker**

Add these types to `cameras.py`:

```python
@dataclass(frozen=True)
class WorkerSnapshot:
    sequence: int
    pair: FramePair | None
    left_detection: DetectionResult | None
    right_detection: DetectionResult | None
    error: str | None


class CameraWorker:
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
        self._stop_event = threading.Event()
        self._retry_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._pair: FramePair | None = None
        self._left_detection: DetectionResult | None = None
        self._right_detection: DetectionResult | None = None
        self._error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise CameraError("camera worker already started")
        self._retry_event.set()
        self._thread = threading.Thread(target=self._run, name="stereo-camera", daemon=True)
        self._thread.start()

    def snapshot(self) -> WorkerSnapshot:
        with self._condition:
            pair = self._pair
            copied = None
            if pair is not None:
                copied = FramePair(
                    pair.left.copy(),
                    pair.right.copy(),
                    pair.left_timestamp_ns,
                    pair.right_timestamp_ns,
                )
            return WorkerSnapshot(
                self._sequence,
                copied,
                self._left_detection,
                self._right_detection,
                self._error,
            )

    def retry(self) -> None:
        self._retry_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._retry_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._thread is not None and self._thread.is_alive():
            raise CameraError("camera worker did not stop within two seconds")

    def _publish_error(self, message: str) -> None:
        with self._condition:
            self._pair = None
            self._left_detection = None
            self._right_detection = None
            self._error = message
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._retry_event.wait()
            self._retry_event.clear()
            if self._stop_event.is_set():
                break
            source = self._source_factory()
            try:
                source.open()
                while not self._stop_event.is_set() and not self._retry_event.is_set():
                    pair = source.read()
                    left_result = detect_checkerboard(pair.left, self._board, self._quality)
                    right_result = detect_checkerboard(pair.right, self._board, self._quality)
                    with self._condition:
                        self._sequence += 1
                        self._pair = pair
                        self._left_detection = left_result
                        self._right_detection = right_result
                        self._error = None
                        self._condition.notify_all()
            except Exception as exc:
                self._publish_error(str(exc))
            finally:
                source.close()
```

The complete file imports `threading`, checkerboard/quality configuration, and detector types. The worker owns the `PairSource` on one daemon thread, keeps only the latest complete pair and results, and records a stable error after failures. Browser threads never call the cameras directly.

- [ ] **Step 5: Test worker reconnect and shutdown behavior**

Use a fake source that fails on a chosen read. Verify the worker disables capture, preserves its error, closes once, can explicitly retry with a fresh source, and exits within two seconds after `stop()`.

- [ ] **Step 6: Run camera tests and commit**

```bash
python -m pytest tools/stereo_calibration/tests/test_cameras.py -v
python -m pytest tools/stereo_calibration/tests -q
git add tools/stereo_calibration/cameras.py tools/stereo_calibration/tests/test_cameras.py
git commit -m "feat: add synchronized dual-camera acquisition"
```

Expected: fake-camera tests pass without accessing `/dev/video*`.

---

### Task 6: Application Service and Loopback Web Interface

**Files:**
- Create: `tools/stereo_calibration/service.py`
- Create: `tools/stereo_calibration/web.py`
- Create: `tools/stereo_calibration/static/index.html`
- Create: `tools/stereo_calibration/tests/test_web.py`

- [ ] **Step 1: Write service-action and HTTP tests**

Use a fake camera worker with one saveable snapshot. Start the HTTP server on `127.0.0.1` and port `0` in a test thread. Verify:

```python
status, body = request_json(server, "GET", "/api/status")
assert status == 200
assert body["left"]["corner_count"] == 54
assert body["right"]["corner_count"] == 54
assert body["can_capture"] is True

status, body = request_json(server, "POST", "/api/capture")
assert status == 201
assert body["pair_id"] == 1

status, headers, jpeg = request_bytes(server, "GET", "/api/preview.jpg")
assert status == 200
assert headers["Content-Type"] == "image/jpeg"
assert jpeg.startswith(b"\xff\xd8")
```

Also test `409 Conflict` when the checkerboard is not saveable, `404` for unknown paths, and rejection when the request `Origin` is not localhost.

- [ ] **Step 2: Run the tests and verify missing-module failures**

```bash
python -m pytest tools/stereo_calibration/tests/test_web.py -v
```

Expected: import fails for `service.py` or `web.py`.

- [ ] **Step 3: Implement `CalibrationService`**

The service is the only application layer allowed to combine worker, detector, session, and calibration engine:

```python
class CalibrationService:
    """Thread-safe application facade used by every HTTP request handler."""
```

Implement these exact methods on the class:

- `status(self) -> dict[str, Any]`: copy one worker snapshot and return camera health, both quality results, `can_capture`, active-pair count, target 30, session ID, and calibration-job state.
- `preview_jpeg(self) -> bytes`: copy one worker snapshot, draw both corner sets, logical-camera labels, rotation status, rejection reasons, and a divider, then return `cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])` bytes or raise `ServiceError("preview encoding failed")`.
- `capture(self) -> SavedPair`: require a complete snapshot and both `DetectionResult.saveable` values; pass those exact images to `SessionStore.save_pair` with both timestamps and both corner arrays converted with `.reshape(-1, 2).tolist()`.
- `reject_last(self) -> SavedPair`: call `SessionStore.reject_last("operator requested deletion")`.
- `start_calibration(self) -> str`: under the job lock, reject a second running job, set state to `running`, and launch one daemon thread that calls `load_session_observations`, validates the minimum pair count, calls `calibrate_observations`, then `write_calibration_run`, and changes state to `pass`, `warning`, `fail`, or `error` from the resulting `report.json`.
- `retry_cameras(self) -> None`: call the worker's explicit `retry()` method and leave saved session data unchanged.

Calibration runs on one background job thread. No HTTP request thread reads a camera or writes calibration arrays directly.

- [ ] **Step 4: Implement the loopback-only standard HTTP server**

Provide exact routes:

```text
GET  /                       static/index.html
GET  /api/status             JSON status and progress
GET  /api/preview.jpg        annotated side-by-side JPEG
POST /api/capture            save current valid pair
POST /api/reject-last        move the most recent active pair to rejected/
POST /api/calibrate          start one background calibration run
POST /api/retry-cameras      explicitly reopen both camera paths
```

Use `ThreadingHTTPServer`. Refuse configuration hosts other than `127.0.0.1`. Set `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, and JSON UTF-8 content types. At server construction, derive the selected port from `server.server_address[1]`; accept POSTs only when `Origin` is absent or equals `http://localhost:{selected_port}` or `http://127.0.0.1:{selected_port}`.

- [ ] **Step 5: Build the complete capture page**

`static/index.html` must contain:

- one side-by-side preview image refreshed with a cache-busting query every 200 ms;
- left and right camera cards showing device status, corner count, sharpness, saturation, and rejection reasons;
- session identifier, valid-pair count, and progress `N / 30`;
- enabled **保存这一组** only when `can_capture` is true;
- **移除上一组**, **重新连接相机**, and **开始标定** controls;
- calibration status and a link-like path display for the newest report;
- a Chinese sampling hint that cycles through center, four corners, opposing tilts, near, and far positions;
- an explicit message that the checkerboard must stop moving before capture.

All requests use relative URLs and plain `fetch`; do not add a JavaScript framework or external CDN dependency.

- [ ] **Step 6: Run web tests and inspect a fake-camera page locally**

```bash
python -m pytest tools/stereo_calibration/tests/test_web.py -v
python -m pytest tools/stereo_calibration/tests -q
```

Start the server with a fake source fixture on an ephemeral port and use `curl` to verify `/api/status` and `/api/preview.jpg`. Expected: HTTP 200 and a decodable JPEG.

- [ ] **Step 7: Commit the browser workflow**

```bash
git add tools/stereo_calibration/service.py tools/stereo_calibration/web.py \
  tools/stereo_calibration/static/index.html tools/stereo_calibration/tests/test_web.py
git commit -m "feat: add guided stereo capture web interface"
```

---

### Task 7: CLI, Documentation, and Offline End-to-End Verification

**Files:**
- Create: `tools/stereo_calibration/main.py`
- Create: `tools/stereo_calibration/README.md`
- Create: `tools/stereo_calibration/tests/test_main.py`

- [ ] **Step 1: Write failing CLI argument tests**

Create `test_main.py` and expose a testable parser:

```python
from dataclasses import replace
from pathlib import Path

import pytest

from tools.stereo_calibration.config import AppConfig, ConfigError
from tools.stereo_calibration.main import build_parser, prepare_session
from tools.stereo_calibration.tests.test_config import valid_payload


def test_cli_defaults_to_example_config() -> None:
    args = build_parser().parse_args([])
    assert args.config.name == "example_config.json"
    assert args.session is None


def test_cli_can_resume_named_session() -> None:
    args = build_parser().parse_args(["--session", "20260826-dry-run"])
    assert args.session == "20260826-dry-run"


def test_resume_refuses_changed_effective_configuration(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_payload(tmp_path))
    session = prepare_session(config, "20260826-dry-run")
    changed = replace(config, baseline_reference_mm=config.baseline_reference_mm + 10.0)
    with pytest.raises(ConfigError, match="does not match the session snapshot"):
        prepare_session(changed, session.session_dir.name)
```

- [ ] **Step 2: Implement process lifecycle and signal-safe shutdown**

Implement these testable setup functions before wiring process signals:

```python
def build_parser() -> argparse.ArgumentParser:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Guided stereo camera calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=package_dir / "example_config.json",
    )
    parser.add_argument("--session")
    parser.add_argument("--check-cameras", action="store_true")
    return parser


def prepare_session(config: AppConfig, session_id: str) -> SessionStore:
    session_dir = config.data_root / session_id
    snapshot_path = session_dir / "config_snapshot.json"
    effective = config.to_mapping()
    if session_dir.exists():
        if not snapshot_path.is_file():
            raise ConfigError("existing session has no configuration snapshot")
        saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if saved != effective:
            raise ConfigError("configuration does not match the session snapshot")
        return SessionStore.open(session_dir)
    store = SessionStore.create(config.data_root, session_id)
    temporary = snapshot_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(snapshot_path)
    return store
```

`main.py` must:

1. parse `--config`, optional `--session`, and `--check-cameras`;
2. load and validate `AppConfig`;
3. create a timestamped session or explicitly resume the named session;
4. atomically write the effective `config.to_mapping()` as `config_snapshot.json` for a new session; on resume, require exact equality with that snapshot and refuse mismatched camera, board, quality, control, or baseline values;
5. start one `CameraWorker`;
6. construct `CalibrationService` and `ThreadingHTTPServer`;
7. print the Ubuntu URL, SSH tunnel command, session directory, and camera logical mapping;
8. handle `SIGINT`/`SIGTERM`, stop the server and worker, and close both camera devices;
9. return a nonzero exit code for configuration or initial camera errors.

The supported invocation is:

```bash
python -m tools.stereo_calibration.main \
  --config tools/stereo_calibration/example_config.json
```

- [ ] **Step 3: Write the beginner operation guide**

`README.md` must include exact sections:

1. **What this tool does and does not do**;
2. **Physical checklist**: rigid cameras, measured 35 mm squares, measured baseline, fixed USB ports;
3. **Ubuntu environment check**:

   ```bash
   source /home/mzq/miniconda3/etc/profile.d/conda.sh
   conda activate tuinadex_hw
   python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
   ```

4. **Camera preflight** with `ls -l` for both approved by-paths;
5. **Start command** and expected startup output;
6. **Mac SSH tunnel** using `mzq.local` first and the current IP only as a fallback;
7. **Three-pair dry run**;
8. **Thirty-pair pose checklist**;
9. **Quality thresholds and report interpretation**;
10. **Resume, camera-retry, and safe shutdown instructions**;
11. **Recalibration rule** when either camera moves;
12. **USB non-synchronization limitation** for later moving subjects.

- [ ] **Step 4: Run offline end-to-end checks**

```bash
python -m compileall tools/stereo_calibration
python -m pytest tools/stereo_calibration/tests -v
git diff --check
```

Expected: compile succeeds, all tests pass, and `git diff --check` is silent.

- [ ] **Step 5: Commit the runnable offline tool**

```bash
git add tools/stereo_calibration/main.py tools/stereo_calibration/README.md \
  tools/stereo_calibration/tests
git commit -m "docs: add stereo calibration operating workflow"
```

---

### Task 8: Ubuntu Laptop Deployment and Three-Pair Hardware Dry Run

**Files:**
- Runtime-only: `data/stereo_calibration/20260826-dry-run/`
- No tracked source changes unless a verified hardware bug requires a separate TDD fix.

- [ ] **Step 1: Verify repository and environment before mutation**

From the Mac:

```bash
ssh mzq@113.54.211.57
```

On Ubuntu:

```bash
cd ~/projects/TuinaDex
git status --short --branch
source /home/mzq/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
df -h ~/projects
python -c "import cv2, numpy; print('opencv:', cv2.__version__); print('numpy:', numpy.__version__)"
python -m pytest --version
```

Expected: the repository has no unexplained changes, disk has at least 2 GB free, OpenCV and NumPy import, and pytest is available. Stop and inspect rather than switching branches if the repository is dirty.

- [ ] **Step 2: Push and check out the feature branch without merging main**

From the isolated Mac worktree used for implementation:

```bash
git push -u origin codex/stereo-calibration
```

On Ubuntu:

```bash
cd ~/projects/TuinaDex
git fetch origin codex/stereo-calibration
git switch --create codex/stereo-calibration --track origin/codex/stereo-calibration
git submodule update --init --recursive
```

If the local branch already exists, use `git switch codex/stereo-calibration` followed by `git pull --ff-only`. Do not merge into `main`.

- [ ] **Step 3: Re-run all offline tests on Ubuntu**

```bash
source /home/mzq/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
cd ~/projects/TuinaDex
python -m pytest tools/stereo_calibration/tests -v
```

Expected: all non-hardware tests pass under the actual Ubuntu/OpenCV build.

- [ ] **Step 4: Verify stable camera paths and actual formats**

```bash
ls -l \
  /dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0 \
  /dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0

v4l2-ctl --device=/dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0 \
  --list-formats-ext
v4l2-ctl --device=/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0 \
  --list-formats-ext
```

Expected: both symlinks resolve and both advertise MJPEG 1280×960 at 30 FPS.

- [ ] **Step 5: Measure and record the exact baseline**

Measure lens optical-center to lens optical-center to the nearest millimetre. Create a user-local configuration and edit `baseline_reference_mm` to that measured number; leave the committed example unchanged:

```bash
mkdir -p ~/.config/tuinadex
cp tools/stereo_calibration/example_config.json \
  ~/.config/tuinadex/stereo-dry-run.json
nano ~/.config/tuinadex/stereo-dry-run.json
```

The initial manual controls are exposure 100, gain 32, and white balance 4600 K. If the browser preview is too dark or bright, stop the service, change only `exposure_time_absolute` or `gain` in this user-local file, and restart the same dry-run session. Apply identical values to both cameras through the single shared control mapping.

- [ ] **Step 6: Start the Ubuntu service**

```bash
source /home/mzq/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
cd ~/projects/TuinaDex
python -m tools.stereo_calibration.main \
  --config ~/.config/tuinadex/stereo-dry-run.json \
  --session 20260826-dry-run
```

Expected: both logical devices open at 1280×960, camera 2 reports 180-degree normalization, and the service listens only on `127.0.0.1:8765`.

- [ ] **Step 7: Open the Mac tunnel and complete the dry run**

In a second Mac terminal:

```bash
ssh -N -L 8765:127.0.0.1:8765 mzq@113.54.211.57
```

Open `http://localhost:8765`. Capture exactly three trial pairs: center, upper-left tilted, and lower-right oppositely tilted. Verify both camera panels show 54/54, the right image is upright, no pair is overwritten, and stopping/restarting with the same session resumes at pair 4.

Once the preview is clear and not severely overexposed, freeze the exact measured baseline and chosen V4L2 values for the formal session:

```bash
cp ~/.config/tuinadex/stereo-dry-run.json \
  ~/.config/tuinadex/stereo-formal.json
```

- [ ] **Step 8: Diagnose hardware failures systematically**

If any hardware step fails, do not change thresholds or camera mapping immediately. Record the exact failing command, browser status JSON, device symlink resolution, and one left/right sample pair. Use `superpowers:systematic-debugging`, reproduce the failure with the smallest focused test, add a failing regression test, then implement and commit the minimal fix.

- [ ] **Step 9: Commit only verified fixes, then push the branch**

```bash
python -m pytest tools/stereo_calibration/tests -v
git status --short
git add tools/stereo_calibration
git commit -m "fix: handle verified stereo camera behavior"
git push origin codex/stereo-calibration
```

Skip the commit if no source change was necessary. Never add `data/stereo_calibration/`.

---

### Task 9: Formal Thirty-Pair Calibration and Acceptance

**Files:**
- Runtime-only: `data/stereo_calibration/20260826-formal/`
- No tracked files unless a regression-tested code fix is required.

- [ ] **Step 1: Freeze the physical and camera settings**

Confirm that neither camera has moved since the dry run, USB paths are unchanged, resolution is 1280×960, logical camera 2 is normalized 180 degrees, and both cameras use the same frozen exposure/gain/white-balance settings.

Start a new formal session with the frozen user-local configuration:

```bash
source /home/mzq/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
cd ~/projects/TuinaDex
python -m tools.stereo_calibration.main \
  --config ~/.config/tuinadex/stereo-formal.json \
  --session 20260826-formal
```

- [ ] **Step 2: Capture thirty distributed pairs**

Use the browser checklist to record:

- 12 center/edge/corner positions;
- 6 left/right tilts;
- 6 up/down tilts;
- 3 near pairs at 550–650 mm;
- 3 far pairs at 900–1000 mm.

Keep the board still, fully visible, and detected 54/54 in both cameras before every click.

- [ ] **Step 3: Run calibration from the browser**

Expected artifacts:

```text
stereo_calibration.npz
stereo_calibration.yaml
report.json
report.md
rectified_previews/*.png
```

- [ ] **Step 4: Apply the numerical acceptance gates**

Accept only when:

- at least 18 pairs are valid and approximately 25 remain after explicit review;
- left, right, and stereo RMS are all at or below 1.5 pixels, with values below 1.0 preferred;
- rectified vertical median error is below 1.0 pixel;
- rectified vertical 95th-percentile error is below 2.0 pixels;
- estimated baseline is within 15% of the directly measured reference.

If a gate fails, inspect the listed outlier IDs, explicitly reject or recapture those poses, and create a new calibration run. Do not overwrite or silently delete the failed run.

- [ ] **Step 5: Visually inspect rectified previews**

Open at least three preview pairs. Corresponding checkerboard corners must lie on the same horizontal guide lines across left and right images. Record the accepted result-directory path.

- [ ] **Step 6: Run final regression and repository checks**

```bash
python -m pytest tools/stereo_calibration/tests -v
python -m compileall tools/stereo_calibration
git status --short --branch
```

Expected: all tests pass, compilation succeeds, and calibration runtime data is absent from Git status.

- [ ] **Step 7: Stop at the approved phase boundary**

Report the accepted calibration directory, exact metrics, measured and estimated baseline, and rectified-preview result. Do not begin disparity, dense point-cloud, LeRobot, arm, or dexterous-hand changes until the point-cloud design is separately approved.
