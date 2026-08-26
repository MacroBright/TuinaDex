"""Tests for validated stereo calibration configuration."""

import json
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
