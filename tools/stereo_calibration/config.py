"""Validated, immutable configuration for guided stereo calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a stereo calibration configuration is invalid."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _required(data: Mapping[str, Any], key: str, label: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ConfigError(f"{label} is missing required field {key!r}") from exc


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    return value


def _require_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"{label} must be an integer")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ConfigError(f"{label} must be a finite number")
    return float(value)


def _require_device_path(value: Any, label: str) -> str:
    device = _require_string(value, label)
    base = PurePosixPath("/dev/v4l/by-path")
    candidate = PurePosixPath(device)
    if not candidate.is_absolute() or candidate.parent != base or not candidate.name:
        raise ConfigError(f"{label} must be a direct child of /dev/v4l/by-path")
    return device


@dataclass(frozen=True)
class CameraConfig:
    """A logical camera selected by a stable V4L2 by-path identifier."""

    name: str
    device: str
    rotation_degrees: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraConfig":
        source = _require_mapping(data, "camera")
        camera = cls(
            name=_require_string(_required(source, "name", "camera"), "camera name"),
            device=_require_device_path(
                _required(source, "device", "camera"), "camera device"
            ),
            rotation_degrees=_require_integer(
                _required(source, "rotation_degrees", "camera"), "camera rotation"
            ),
        )
        if not camera.name:
            raise ConfigError("camera name must not be empty")
        if camera.rotation_degrees not in (0, 90, 180, 270):
            raise ConfigError("camera rotation must be 0, 90, 180, or 270 degrees")
        return camera


@dataclass(frozen=True)
class CaptureConfig:
    width: int
    height: int
    fps: int
    fourcc: str
    v4l2_controls: Mapping[str, int]

    def __post_init__(self) -> None:
        # A frozen dataclass does not freeze a dictionary supplied by a caller.
        object.__setattr__(self, "v4l2_controls", MappingProxyType(dict(self.v4l2_controls)))

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

    @property
    def logical_image_size(self) -> tuple[int, int]:
        """Return the shared post-rotation width and height."""
        if self.left.rotation_degrees in (90, 270):
            return self.capture.height, self.capture.width
        return self.capture.image_size

    @classmethod
    def from_json(cls, path: Path | str) -> "AppConfig":
        def reject_constant(value: str) -> Any:
            raise ConfigError(f"JSON constant {value!r} is not a finite number")

        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"could not read configuration {path!s}") from exc
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        source = _require_mapping(data, "application configuration")
        left = CameraConfig.from_mapping(_required(source, "left", "application configuration"))
        right = CameraConfig.from_mapping(_required(source, "right", "application configuration"))
        if left.device == right.device:
            raise ConfigError("left and right cameras must use different devices")

        capture_source = _require_mapping(
            _required(source, "capture", "application configuration"), "capture"
        )
        controls_source = _require_mapping(
            _required(capture_source, "v4l2_controls", "capture"), "capture.v4l2_controls"
        )
        controls = {
            _require_string(name, "v4l2 control name"): _require_integer(
                value, f"v4l2 control {name!r}"
            )
            for name, value in controls_source.items()
        }
        capture = CaptureConfig(
            width=_require_integer(_required(capture_source, "width", "capture"), "capture width"),
            height=_require_integer(
                _required(capture_source, "height", "capture"), "capture height"
            ),
            fps=_require_integer(_required(capture_source, "fps", "capture"), "capture fps"),
            fourcc=_require_string(_required(capture_source, "fourcc", "capture"), "capture fourcc"),
            v4l2_controls=controls,
        )
        checkerboard_source = _require_mapping(
            _required(source, "checkerboard", "application configuration"), "checkerboard"
        )
        checkerboard = CheckerboardConfig(
            columns=_require_integer(
                _required(checkerboard_source, "columns", "checkerboard"), "checkerboard columns"
            ),
            rows=_require_integer(
                _required(checkerboard_source, "rows", "checkerboard"), "checkerboard rows"
            ),
            square_size_mm=_require_finite_number(
                _required(checkerboard_source, "square_size_mm", "checkerboard"),
                "checkerboard square size",
            ),
        )
        quality_source = _require_mapping(
            _required(source, "quality", "application configuration"), "quality"
        )
        quality = QualityConfig(
            edge_margin_px=_require_integer(
                _required(quality_source, "edge_margin_px", "quality"), "quality edge margin"
            ),
            min_laplacian_variance=_require_finite_number(
                _required(quality_source, "min_laplacian_variance", "quality"),
                "minimum Laplacian variance",
            ),
            max_saturated_fraction=_require_finite_number(
                _required(quality_source, "max_saturated_fraction", "quality"),
                "maximum saturated fraction",
            ),
        )
        web_source = _require_mapping(
            _required(source, "web", "application configuration"), "web"
        )
        web = WebConfig(
            host=_require_string(_required(web_source, "host", "web"), "web host"),
            port=_require_integer(_required(web_source, "port", "web"), "web port"),
        )
        config = cls(
            left=left,
            right=right,
            capture=capture,
            checkerboard=checkerboard,
            quality=quality,
            data_root=Path(
                _require_string(
                    _required(source, "data_root", "application configuration"), "data root"
                )
            ).expanduser(),
            web=web,
            minimum_pairs=_require_integer(
                _required(source, "minimum_pairs", "application configuration"), "minimum pairs"
            ),
            target_pairs=_require_integer(
                _required(source, "target_pairs", "application configuration"), "target pairs"
            ),
            baseline_reference_mm=_require_finite_number(
                _required(source, "baseline_reference_mm", "application configuration"),
                "baseline reference",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate all cross-field and range constraints."""
        if self.left.device == self.right.device:
            raise ConfigError("left and right cameras must use different devices")
        for camera in (self.left, self.right):
            if not _require_string(camera.name, "camera name"):
                raise ConfigError("camera name must not be empty")
            _require_device_path(camera.device, "camera device")
            rotation = _require_integer(camera.rotation_degrees, "camera rotation")
            if rotation not in (0, 90, 180, 270):
                raise ConfigError("camera rotation must be 0, 90, 180, or 270 degrees")

        left_is_quarter_turn = self.left.rotation_degrees in (90, 270)
        right_is_quarter_turn = self.right.rotation_degrees in (90, 270)
        if left_is_quarter_turn != right_is_quarter_turn:
            raise ConfigError("left and right logical image dimensions must match")

        width = _require_integer(self.capture.width, "capture width")
        height = _require_integer(self.capture.height, "capture height")
        fps = _require_integer(self.capture.fps, "capture fps")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ConfigError("capture width, height, and fps must be positive")
        fourcc = _require_string(self.capture.fourcc, "capture fourcc")
        if len(fourcc) != 4:
            raise ConfigError("capture fourcc must contain four characters")
        controls = _require_mapping(self.capture.v4l2_controls, "capture.v4l2_controls")
        if not controls:
            raise ConfigError("v4l2 controls must be a non-empty string-to-integer mapping")
        for name, value in controls.items():
            if not _require_string(name, "v4l2 control name"):
                raise ConfigError("v4l2 control names must not be empty")
            _require_integer(value, f"v4l2 control {name!r}")

        columns = _require_integer(self.checkerboard.columns, "checkerboard columns")
        rows = _require_integer(self.checkerboard.rows, "checkerboard rows")
        square_size_mm = _require_finite_number(
            self.checkerboard.square_size_mm, "checkerboard square size"
        )
        if columns < 2 or rows < 2:
            raise ConfigError("checkerboard dimensions must be at least 2x2")
        if square_size_mm <= 0:
            raise ConfigError("checkerboard square size must be positive")
        edge_margin_px = _require_integer(self.quality.edge_margin_px, "quality edge margin")
        min_laplacian_variance = _require_finite_number(
            self.quality.min_laplacian_variance, "minimum Laplacian variance"
        )
        max_saturated_fraction = _require_finite_number(
            self.quality.max_saturated_fraction, "maximum saturated fraction"
        )
        if edge_margin_px < 0 or min_laplacian_variance < 0:
            raise ConfigError("quality thresholds must be non-negative")
        if not 0.0 < max_saturated_fraction < 1.0:
            raise ConfigError("max saturated fraction must be between 0 and 1")
        minimum_pairs = _require_integer(self.minimum_pairs, "minimum pairs")
        target_pairs = _require_integer(self.target_pairs, "target pairs")
        if minimum_pairs < 3:
            raise ConfigError("minimum pairs must be at least 3")
        if target_pairs < minimum_pairs:
            raise ConfigError("target pairs must be greater than or equal to minimum pairs")
        baseline_reference_mm = _require_finite_number(
            self.baseline_reference_mm, "baseline reference"
        )
        if baseline_reference_mm <= 0:
            raise ConfigError("baseline reference must be positive")
        host = _require_string(self.web.host, "web host")
        port = _require_integer(self.web.port, "web port")
        if host != "127.0.0.1":
            raise ConfigError("web server must bind to 127.0.0.1")
        if port <= 0 or port > 65535:
            raise ConfigError("web port must be between 1 and 65535")

    def to_mapping(self) -> dict[str, Any]:
        """Return the effective configuration as a JSON-serializable mapping."""
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
