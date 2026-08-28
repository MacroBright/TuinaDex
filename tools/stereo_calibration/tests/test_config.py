"""Tests for validated stereo calibration configuration."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tools.stereo_calibration.config import AppConfig, ConfigError


LEFT_CAMERA_PATH = "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0"
RIGHT_CAMERA_PATH = "/dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0"


def valid_payload(tmp_path: Path) -> dict:
    """Return the approved application configuration with a temporary data root."""
    return {
        "left": {
            "name": "usb_port_1",
            "device": LEFT_CAMERA_PATH,
            "rotation_degrees": 0,
        },
        "right": {
            "name": "usb_port_2",
            "device": RIGHT_CAMERA_PATH,
            "rotation_degrees": 180,
        },
        "data_root": str(tmp_path / "stereo-calibration"),
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
        "web": {"host": "127.0.0.1", "port": 8765},
        "minimum_pairs": 18,
        "target_pairs": 30,
        "baseline_reference_mm": 200.0,
    }


# Keep the descriptive name used by this task while exposing the plan's helper
# name for downstream tests that reuse the approved payload.
valid_mapping = valid_payload


def test_loads_approved_camera_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_mapping(tmp_path)), encoding="utf-8")
    config = AppConfig.from_json(path)

    assert config.left.device == LEFT_CAMERA_PATH
    assert config.right.device == RIGHT_CAMERA_PATH


def test_right_camera_rotation_is_180(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_mapping(tmp_path))

    assert config.right.rotation_degrees == 180


def test_quarter_turn_cameras_swap_the_logical_image_size(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["left"]["rotation_degrees"] = 270
    mapping["right"]["rotation_degrees"] = 90

    config = AppConfig.from_mapping(mapping)

    assert config.capture.image_size == (1280, 960)
    assert config.logical_image_size == (960, 1280)


def test_mixed_quarter_turn_parity_is_rejected(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["left"]["rotation_degrees"] = 90
    mapping["right"]["rotation_degrees"] = 180

    with pytest.raises(ConfigError, match="logical image dimensions"):
        AppConfig.from_mapping(mapping)


@pytest.mark.parametrize("rotation", [-90, 45, 360])
def test_unsupported_integer_camera_rotation_is_rejected(
    tmp_path: Path, rotation: int
) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["right"]["rotation_degrees"] = rotation

    with pytest.raises(ConfigError, match="rotation"):
        AppConfig.from_mapping(mapping)


def test_fractional_camera_rotation_is_rejected(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["right"]["rotation_degrees"] = 180.9

    with pytest.raises(ConfigError, match="rotation"):
        AppConfig.from_mapping(mapping)


@pytest.mark.parametrize("rotation", [0.0, 180.0, True, 180.9, "180"])
def test_non_integer_camera_rotation_is_rejected(tmp_path: Path, rotation: object) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["right"]["rotation_degrees"] = rotation

    with pytest.raises(ConfigError, match="rotation"):
        AppConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("capture", "width", 1280.9, "width"),
        ("capture", "width", "1280", "width"),
        ("capture", "width", True, "width"),
        ("capture", "fourcc", 1234, "fourcc"),
        ("capture", "v4l2_controls", {"gain": True}, "v4l2"),
    ],
)
def test_primitive_types_are_not_coerced(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    mapping = valid_mapping(tmp_path)
    mapping[section][field] = value

    with pytest.raises(ConfigError, match=message):
        AppConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("checkerboard", "square_size_mm"),
        ("quality", "min_laplacian_variance"),
        ("quality", "max_saturated_fraction"),
        ("app", "baseline_reference_mm"),
    ],
)
def test_non_finite_physical_values_are_rejected(
    tmp_path: Path, section: str, field: str
) -> None:
    mapping = valid_mapping(tmp_path)
    target = mapping if section == "app" else mapping[section]
    target[field] = math.nan

    with pytest.raises(ConfigError):
        AppConfig.from_mapping(mapping)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_non_finite_constants_are_rejected(tmp_path: Path, constant: str) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["checkerboard"]["square_size_mm"] = float(constant)
    path = tmp_path / "non-finite.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ConfigError):
        AppConfig.from_json(path)


@pytest.mark.parametrize("device", ["/dev/v4l/by-path/../video4", "/dev/v4l/by-path/"])
def test_camera_device_must_be_a_direct_by_path_child(tmp_path: Path, device: str) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["left"]["device"] = device

    with pytest.raises(ConfigError, match=r"/dev/v4l/by-path"):
        AppConfig.from_mapping(mapping)


def test_normal_camera_device_path_is_accepted(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_mapping(tmp_path))

    assert config.left.device == LEFT_CAMERA_PATH


def test_loads_committed_example_configuration() -> None:
    path = Path(__file__).parents[1] / "example_config.json"
    config = AppConfig.from_json(path)

    assert config.left.device == LEFT_CAMERA_PATH
    assert config.right.device == RIGHT_CAMERA_PATH
    assert config.right.rotation_degrees == 180
    assert config.capture.image_size == (1280, 960)
    assert config.capture.fps == 30
    assert config.capture.fourcc == "MJPG"
    assert dict(config.capture.v4l2_controls) == {
        "auto_exposure": 1,
        "exposure_time_absolute": 100,
        "gain": 32,
        "white_balance_automatic": 0,
        "white_balance_temperature": 4600,
    }
    assert config.checkerboard.pattern_size == (9, 6)
    assert config.checkerboard.square_size_mm == 35.0
    assert config.quality.edge_margin_px == 12
    assert config.quality.min_laplacian_variance == 60.0
    assert config.quality.max_saturated_fraction == 0.45
    assert config.web.host == "127.0.0.1"
    assert config.web.port == 8765
    assert config.minimum_pairs == 18
    assert config.target_pairs == 30
    assert config.baseline_reference_mm == 200.0


def test_controls_are_copied_and_immutable(tmp_path: Path) -> None:
    controls = {"gain": 32}
    mapping = valid_mapping(tmp_path)
    mapping["capture"]["v4l2_controls"] = controls
    config = AppConfig.from_mapping(mapping)

    controls["gain"] = 99
    assert config.capture.v4l2_controls["gain"] == 32
    with pytest.raises(TypeError):
        config.capture.v4l2_controls["gain"] = 99


def test_validate_rejects_empty_camera_name_on_direct_construction(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_payload(tmp_path))
    invalid = replace(config, left=replace(config.left, name=""))

    with pytest.raises(ConfigError, match="camera name"):
        invalid.validate()


def test_checkerboard_corner_count_is_54(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_mapping(tmp_path))

    assert config.checkerboard.corner_count == 54


def test_capture_image_size_is_1280_by_960(tmp_path: Path) -> None:
    config = AppConfig.from_mapping(valid_mapping(tmp_path))

    assert config.capture.image_size == (1280, 960)


def test_transient_video_device_is_rejected(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["left"]["device"] = "/dev/video4"

    with pytest.raises(ConfigError, match=r"/dev/v4l/by-path"):
        AppConfig.from_mapping(mapping)


def test_duplicate_camera_paths_are_rejected(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["right"]["device"] = mapping["left"]["device"]

    with pytest.raises(ConfigError, match="different devices"):
        AppConfig.from_mapping(mapping)


def test_effective_mapping_round_trips_with_expanded_data_root(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    mapping["data_root"] = "~/tuinadex-stereo-calibration"

    config = AppConfig.from_mapping(mapping)
    effective = config.to_mapping()
    round_trip = AppConfig.from_mapping(effective)

    assert effective["data_root"] == str(Path("~/tuinadex-stereo-calibration").expanduser())
    assert round_trip.to_mapping() == effective
    json.dumps(effective)
