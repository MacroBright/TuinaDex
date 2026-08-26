"""Validated, immutable configuration for guided stereo calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a stereo calibration configuration is invalid."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _required(data: Mapping[str, Any], key: str, label: str) -> Any:
    try:
        return data[key]
    except KeyError as exc:
        raise ConfigError(f"{label} is missing required field {key!r}") from exc


def _rotation_value(value: Any) -> int:
    """Parse a rotation without truncating fractional or boolean values."""
    if isinstance(value, bool) or not isinstance(value, Real) or value not in (0, 180):
        raise ConfigError("camera rotation must be numeric and exactly 0 or 180 degrees")
    return int(value)


@dataclass(frozen=True)
class CameraConfig:
    """A logical camera selected by a stable V4L2 by-path identifier."""

    name: str
    device: str
    rotation_degrees: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraConfig":
        source = _mapping(data, "camera")
        try:
            camera = cls(
                name=str(_required(source, "name", "camera")),
                device=str(_required(source, "device", "camera")),
                rotation_degrees=_rotation_value(
                    _required(source, "rotation_degrees", "camera")
                ),
            )
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError("camera name, device, and rotation_degrees are invalid") from exc
        if not camera.name:
            raise ConfigError("camera name must not be empty")
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

    @classmethod
    def from_json(cls, path: Path | str) -> "AppConfig":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"could not read configuration {path!s}") from exc
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        source = _mapping(data, "application configuration")
        left = CameraConfig.from_mapping(_required(source, "left", "application configuration"))
        right = CameraConfig.from_mapping(_required(source, "right", "application configuration"))
        if left.device == right.device:
            raise ConfigError("left and right cameras must use different devices")

        capture_source = _mapping(
            _required(source, "capture", "application configuration"), "capture"
        )
        try:
            capture = CaptureConfig(
                width=int(_required(capture_source, "width", "capture")),
                height=int(_required(capture_source, "height", "capture")),
                fps=int(_required(capture_source, "fps", "capture")),
                fourcc=str(_required(capture_source, "fourcc", "capture")),
                v4l2_controls=MappingProxyType(
                    dict(
                        _mapping(
                            _required(capture_source, "v4l2_controls", "capture"),
                            "capture.v4l2_controls",
                        )
                    )
                ),
            )
            checkerboard_source = _mapping(
                _required(source, "checkerboard", "application configuration"), "checkerboard"
            )
            checkerboard = CheckerboardConfig(
                columns=int(_required(checkerboard_source, "columns", "checkerboard")),
                rows=int(_required(checkerboard_source, "rows", "checkerboard")),
                square_size_mm=float(
                    _required(checkerboard_source, "square_size_mm", "checkerboard")
                ),
            )
            quality_source = _mapping(
                _required(source, "quality", "application configuration"), "quality"
            )
            quality = QualityConfig(
                edge_margin_px=int(_required(quality_source, "edge_margin_px", "quality")),
                min_laplacian_variance=float(
                    _required(quality_source, "min_laplacian_variance", "quality")
                ),
                max_saturated_fraction=float(
                    _required(quality_source, "max_saturated_fraction", "quality")
                ),
            )
            web_source = _mapping(
                _required(source, "web", "application configuration"), "web"
            )
            web = WebConfig(
                host=str(_required(web_source, "host", "web")),
                port=int(_required(web_source, "port", "web")),
            )
            config = cls(
                left=left,
                right=right,
                capture=capture,
                checkerboard=checkerboard,
                quality=quality,
                data_root=Path(_required(source, "data_root", "application configuration"))
                .expanduser(),
                web=web,
                minimum_pairs=int(_required(source, "minimum_pairs", "application configuration")),
                target_pairs=int(_required(source, "target_pairs", "application configuration")),
                baseline_reference_mm=float(
                    _required(source, "baseline_reference_mm", "application configuration")
                ),
            )
        except ConfigError:
            raise
        except (TypeError, ValueError, OSError) as exc:
            raise ConfigError("application configuration contains invalid values") from exc
        config.validate()
        return config

    def validate(self) -> None:
        """Validate all cross-field and range constraints."""
        if self.left.device == self.right.device:
            raise ConfigError("left and right cameras must use different devices")
        for camera in (self.left, self.right):
            if not camera.device.startswith("/dev/v4l/by-path/"):
                raise ConfigError("camera device must use /dev/v4l/by-path")
            if (
                isinstance(camera.rotation_degrees, bool)
                or not isinstance(camera.rotation_degrees, Real)
                or camera.rotation_degrees not in (0, 180)
            ):
                raise ConfigError("camera rotation must be 0 or 180 degrees")

        if self.capture.width <= 0 or self.capture.height <= 0 or self.capture.fps <= 0:
            raise ConfigError("capture width, height, and fps must be positive")
        if not isinstance(self.capture.fourcc, str) or len(self.capture.fourcc) != 4:
            raise ConfigError("capture fourcc must contain four characters")
        controls = self.capture.v4l2_controls
        if not controls or not all(
            isinstance(name, str) and name and isinstance(value, int) and not isinstance(value, bool)
            for name, value in controls.items()
        ):
            raise ConfigError("v4l2 controls must be a non-empty string-to-integer mapping")

        if self.checkerboard.columns < 2 or self.checkerboard.rows < 2:
            raise ConfigError("checkerboard dimensions must be at least 2x2")
        if self.checkerboard.square_size_mm <= 0:
            raise ConfigError("checkerboard square size must be positive")
        if self.quality.edge_margin_px < 0 or self.quality.min_laplacian_variance < 0:
            raise ConfigError("quality thresholds must be non-negative")
        if not 0.0 < self.quality.max_saturated_fraction < 1.0:
            raise ConfigError("max saturated fraction must be between 0 and 1")
        if self.minimum_pairs < 3:
            raise ConfigError("minimum pairs must be at least 3")
        if self.target_pairs < self.minimum_pairs:
            raise ConfigError("target pairs must be greater than or equal to minimum pairs")
        if self.baseline_reference_mm <= 0:
            raise ConfigError("baseline reference must be positive")
        if self.web.host != "127.0.0.1":
            raise ConfigError("web server must bind to 127.0.0.1")
        if self.web.port <= 0 or self.web.port > 65535:
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
