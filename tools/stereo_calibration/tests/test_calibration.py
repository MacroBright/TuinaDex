"""Behavior tests for stereo calibration mathematics and artifacts."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.stereo_calibration import calibration
from tools.stereo_calibration.calibration import (
    CalibrationResult,
    StereoObservation,
    calibrate_observations,
    load_session_observations,
    write_calibration_run,
)
from tools.stereo_calibration.config import AppConfig
from tools.stereo_calibration.session import SavedPair, SessionStore
from tools.stereo_calibration.tests.helpers import (
    SYNTHETIC_BOARD,
    synthetic_stereo_observations,
)
from tools.stereo_calibration.tests.test_config import valid_payload


@pytest.fixture(scope="module")
def clean_result() -> CalibrationResult:
    return calibrate_observations(
        synthetic_stereo_observations(), image_size=(1280, 960)
    )


def test_synthetic_fixture_contains_exactly_twenty_independent_float32_views() -> None:
    observations = synthetic_stereo_observations()

    assert len(observations) == 20
    assert [observation.pair_id for observation in observations] == list(range(20))
    for observation in observations:
        assert observation.object_points.shape == (54, 3)
        assert observation.left_corners.shape == (54, 1, 2)
        assert observation.right_corners.shape == (54, 1, 2)
        assert observation.object_points.dtype == np.float32
        assert observation.left_corners.dtype == np.float32
        assert observation.right_corners.dtype == np.float32
    assert not np.shares_memory(
        observations[0].object_points, observations[1].object_points
    )


def test_recovers_approximately_200_mm_baseline(clean_result: CalibrationResult) -> None:
    assert clean_result.pair_count == 20
    assert abs(clean_result.baseline_mm - 200.0) < 2.0
    assert clean_result.stereo_rms < 0.1
    assert clean_result.vertical_error_median_px < 0.1
    assert clean_result.vertical_error_p95_px < 0.1
    assert clean_result.map_left_x.dtype == np.float32
    assert clean_result.map_left_y.dtype == np.float32
    assert clean_result.map_right_x.dtype == np.float32
    assert clean_result.map_right_y.dtype == np.float32
    assert clean_result.map_left_x.shape == (960, 1280)


def test_rejects_fewer_than_minimum_pairs() -> None:
    observations = synthetic_stereo_observations()[:17]
    with pytest.raises(ValueError, match="at least 18 valid pairs"):
        calibrate_observations(
            observations, image_size=(1280, 960), minimum_pairs=18
        )


@pytest.mark.parametrize(
    ("image_size", "message"),
    [
        ((0, 960), "positive integer"),
        ((1280.0, 960), "positive integer"),
        ((True, 960), "positive integer"),
        ((1280,), "width and height"),
    ],
)
def test_rejects_invalid_image_size(image_size: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate_observations(  # type: ignore[arg-type]
            synthetic_stereo_observations(), image_size=image_size
        )


def test_rejects_duplicate_and_negative_pair_ids() -> None:
    observations = synthetic_stereo_observations()
    with pytest.raises(ValueError, match="unique"):
        calibrate_observations(
            [observations[0], replace(observations[1], pair_id=0), *observations[2:]],
            image_size=(1280, 960),
        )
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_observations(
            [replace(observations[0], pair_id=-1), *observations[1:]],
            image_size=(1280, 960),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        (
            "object_points",
            np.zeros((54, 3), dtype=np.float64),
            "object_points must have dtype float32",
        ),
        (
            "left_corners",
            np.zeros((54, 2), dtype=np.float32),
            r"left_corners must have shape \(54, 1, 2\)",
        ),
        (
            "right_corners",
            np.zeros((53, 1, 2), dtype=np.float32),
            "consistent point counts",
        ),
    ],
)
def test_rejects_malformed_observation_arrays(
    field: str, bad_value: np.ndarray, message: str
) -> None:
    observations = synthetic_stereo_observations()
    observations[0] = replace(observations[0], **{field: bad_value})

    with pytest.raises(ValueError, match=message):
        calibrate_observations(observations, image_size=(1280, 960))


def test_rejects_nan_observation_values() -> None:
    observations = synthetic_stereo_observations()
    corners = observations[3].left_corners.copy()
    corners[0, 0, 0] = np.nan
    observations[3] = replace(observations[3], left_corners=corners)

    with pytest.raises(ValueError, match=r"pair 3.*left_corners.*finite"):
        calibrate_observations(observations, image_size=(1280, 960))


def _pair_metadata(observation: StereoObservation, image_size: tuple[int, int]) -> dict:
    return {
        "left_corners": observation.left_corners.reshape(-1, 2).tolist(),
        "right_corners": observation.right_corners.reshape(-1, 2).tolist(),
        "image_size": list(image_size),
    }


def test_load_session_skips_bad_metadata_with_pair_specific_diagnostic(
    tmp_path: Path,
) -> None:
    store = SessionStore.create(tmp_path, "load")
    observations = synthetic_stereo_observations()
    image = np.full((960, 1280, 3), 80, dtype=np.uint8)
    first = store.save_pair(image, image, _pair_metadata(observations[0], (1280, 960)))
    malformed = _pair_metadata(observations[1], (1280, 960))
    malformed["left_corners"] = malformed["left_corners"][:-1]
    second = store.save_pair(image, image, malformed)

    loaded, diagnostics = load_session_observations(store, SYNTHETIC_BOARD)

    assert [observation.pair_id for observation in loaded] == [first.pair_id]
    assert diagnostics == [
        {
            "pair_id": second.pair_id,
            "reason": "left_corners must have shape (54, 2)",
        }
    ]
    assert loaded[0].left_corners.shape == (54, 1, 2)
    assert loaded[0].object_points.dtype == np.float32


def test_load_session_checks_saved_image_dimensions_and_continues(
    tmp_path: Path,
) -> None:
    store = SessionStore.create(tmp_path, "dimensions")
    observation = synthetic_stereo_observations()[0]
    left = np.zeros((960, 1280, 3), dtype=np.uint8)
    right = np.zeros((959, 1280, 3), dtype=np.uint8)
    saved = store.save_pair(left, right, _pair_metadata(observation, (1280, 960)))

    loaded, diagnostics = load_session_observations(store, SYNTHETIC_BOARD)

    assert loaded == []
    assert diagnostics == [
        {
            "pair_id": saved.pair_id,
            "reason": "left/right captured image dimensions differ: 1280x960 versus 1280x959",
        }
    ]


def test_load_session_requires_configured_image_dimensions_in_metadata(
    tmp_path: Path,
) -> None:
    store = SessionStore.create(tmp_path, "missing-dimensions")
    observation = synthetic_stereo_observations()[0]
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    metadata = _pair_metadata(observation, (1280, 960))
    del metadata["image_size"]
    saved = store.save_pair(image, image, metadata)

    loaded, diagnostics = load_session_observations(store, SYNTHETIC_BOARD)

    assert loaded == []
    assert diagnostics == [
        {"pair_id": saved.pair_id, "reason": "image_size metadata is missing"}
    ]


def _bounded_metadata(width: int, height: int) -> dict:
    xs = np.linspace(2.0, width - 3.0, SYNTHETIC_BOARD.columns, dtype=np.float32)
    ys = np.linspace(2.0, height - 3.0, SYNTHETIC_BOARD.rows, dtype=np.float32)
    corners = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return {
        "left_corners": corners.tolist(),
        "right_corners": corners.tolist(),
        "image_size": [width, height],
    }


def test_load_session_rejects_cross_pair_size_mismatch_without_poisoning_later_pair(
    tmp_path: Path,
) -> None:
    store = SessionStore.create(tmp_path, "cross-pair-size")
    small = np.zeros((48, 64, 3), dtype=np.uint8)
    large = np.zeros((60, 80, 3), dtype=np.uint8)
    first = store.save_pair(small, small, _bounded_metadata(64, 48))
    mismatched = store.save_pair(large, large, _bounded_metadata(80, 60))
    third = store.save_pair(small, small, _bounded_metadata(64, 48))

    observations, diagnostics = load_session_observations(store, SYNTHETIC_BOARD)

    assert [item.pair_id for item in observations] == [first.pair_id, third.pair_id]
    assert diagnostics == [
        {
            "pair_id": mismatched.pair_id,
            "reason": "captured image dimensions 80x60 differ from session dimensions 64x48",
        }
    ]


def test_load_session_rejects_out_of_bounds_corners_and_continues(
    tmp_path: Path,
) -> None:
    store = SessionStore.create(tmp_path, "corner-bounds")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    first_metadata = _bounded_metadata(64, 48)
    first_metadata["right_corners"][0][0] = 64.0
    invalid = store.save_pair(image, image, first_metadata)
    valid = store.save_pair(image, image, _bounded_metadata(64, 48))

    observations, diagnostics = load_session_observations(store, SYNTHETIC_BOARD)

    assert [item.pair_id for item in observations] == [valid.pair_id]
    assert diagnostics == [
        {
            "pair_id": invalid.pair_id,
            "reason": "right_corners contains coordinates outside image bounds 64x48",
        }
    ]


def _artifact_config(tmp_path: Path, baseline_reference_mm: float = 200.0) -> AppConfig:
    payload = valid_payload(tmp_path)
    payload["capture"]["width"] = 64
    payload["capture"]["height"] = 48
    payload["baseline_reference_mm"] = baseline_reference_mm
    return AppConfig.from_mapping(payload)


def _identity_maps(width: int = 64, height: int = 48) -> tuple[np.ndarray, np.ndarray]:
    map_x, map_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return map_x, map_y


def _artifact_result(
    *,
    left_rms: float = 0.2,
    right_rms: float = 0.3,
    stereo_rms: float = 0.4,
    vertical_median: float = 0.2,
    vertical_p95: float = 0.4,
    baseline_mm: float = 200.0,
    per_pair_errors: dict[int, dict[str, float]] | None = None,
) -> CalibrationResult:
    map_x, map_y = _identity_maps()
    camera_matrix = np.array(
        [[90.0, 0.0, 32.0], [0.0, 90.0, 24.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projection_left = np.column_stack((camera_matrix, np.zeros(3)))
    projection_right = projection_left.copy()
    projection_right[0, 3] = -90.0 * baseline_mm
    errors = per_pair_errors or {
        pair_id: {
            "left_rmse_px": 0.1 * pair_id,
            "right_rmse_px": 0.1 * pair_id,
            "combined_rmse_px": 0.1 * pair_id,
        }
        for pair_id in (1, 2, 3)
    }
    return CalibrationResult(
        pair_count=len(errors),
        left_rms=left_rms,
        right_rms=right_rms,
        stereo_rms=stereo_rms,
        left_camera_matrix=camera_matrix.copy(),
        right_camera_matrix=camera_matrix.copy(),
        left_distortion=np.zeros((1, 5), dtype=np.float64),
        right_distortion=np.zeros((1, 5), dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        translation=np.array([[-baseline_mm], [0.0], [0.0]], dtype=np.float64),
        essential=np.zeros((3, 3), dtype=np.float64),
        fundamental=np.zeros((3, 3), dtype=np.float64),
        rectification_left=np.eye(3, dtype=np.float64),
        rectification_right=np.eye(3, dtype=np.float64),
        projection_left=projection_left,
        projection_right=projection_right,
        disparity_to_depth=np.eye(4, dtype=np.float64),
        map_left_x=map_x.copy(),
        map_left_y=map_y.copy(),
        map_right_x=map_x.copy(),
        map_right_y=map_y.copy(),
        per_pair_errors=errors,
        vertical_error_median_px=vertical_median,
        vertical_error_p95_px=vertical_p95,
    )


def _source_pairs(store: SessionStore, pair_ids: tuple[int, ...] = (1, 2, 3)) -> list[SavedPair]:
    pairs: list[SavedPair] = []
    for pair_id in pair_ids:
        image = np.full((48, 64, 3), 25 * pair_id, dtype=np.uint8)
        left_path = store.pairs_dir / f"pair_{pair_id:04d}_left.png"
        right_path = store.pairs_dir / f"pair_{pair_id:04d}_right.png"
        assert cv2.imwrite(str(left_path), image)
        assert cv2.imwrite(str(right_path), image)
        pairs.append(SavedPair(pair_id, left_path, right_path, {}))
    return pairs


def test_write_run_round_trips_npz_yaml_json_markdown_and_previews(
    tmp_path: Path,
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "artifacts")
    result = _artifact_result()
    sources = _source_pairs(store)

    run_dir = write_calibration_run(store, result, config, sources)

    expected_keys = {
        "left_camera_matrix",
        "right_camera_matrix",
        "left_distortion",
        "right_distortion",
        "rotation",
        "translation",
        "essential",
        "fundamental",
        "rectification_left",
        "rectification_right",
        "projection_left",
        "projection_right",
        "disparity_to_depth",
        "map_left_x",
        "map_left_y",
        "map_right_x",
        "map_right_y",
    }
    with np.load(run_dir / "stereo_calibration.npz") as archive:
        assert expected_keys <= set(archive.files)
        assert archive["translation"].shape == (3, 1)

    storage = cv2.FileStorage(
        str(run_dir / "stereo_calibration.yaml"), cv2.FILE_STORAGE_READ
    )
    try:
        assert storage.isOpened()
        assert storage.getNode("left_camera_matrix").mat().shape == (3, 3)
        assert storage.getNode("projection_right").mat().shape == (3, 4)
        assert int(storage.getNode("pair_count").real()) == 3
    finally:
        storage.release()

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["metrics"]["baseline_mm"] == pytest.approx(200.0)
    assert report["source_pair_ids"] == [1, 2, 3]
    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "标定状态：通过" in markdown
    assert "基线长度" in markdown

    previews = sorted((run_dir / "previews").glob("*.png"))
    assert len(previews) >= 3
    preview = cv2.imread(str(previews[0]))
    assert preview is not None
    assert preview.shape == (48, 128, 3)
    assert np.any(preview[40, :, 1] > 150)


def test_nan_matrix_is_rejected_before_any_artifact_or_lock_is_created(
    tmp_path: Path,
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "nan-matrix")
    sources = _source_pairs(store)
    matrix = _artifact_result().left_camera_matrix.copy()
    matrix[0, 0] = np.nan
    result = replace(_artifact_result(), left_camera_matrix=matrix)

    with pytest.raises(ValueError, match="left_camera_matrix.*finite"):
        write_calibration_run(store, result, config, sources)

    assert list(store.results_dir.iterdir()) == []


def test_result_and_source_ids_must_match_before_artifact_creation(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "id-mismatch")
    sources = _source_pairs(store, (1, 2, 4))

    with pytest.raises(ValueError, match="source pair IDs must exactly match result pair IDs"):
        write_calibration_run(store, _artifact_result(), config, sources)

    assert list(store.results_dir.iterdir()) == []


def test_pair_count_must_match_per_pair_errors_before_artifact_creation(
    tmp_path: Path,
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "count-mismatch")
    result = replace(_artifact_result(), pair_count=4)

    with pytest.raises(ValueError, match="pair_count must equal the per-pair error count"):
        write_calibration_run(store, result, config, _source_pairs(store))

    assert list(store.results_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(float("nan"), "finite"), (-0.1, "non-negative")],
)
def test_invalid_individual_reprojection_error_is_rejected_before_artifacts(
    tmp_path: Path, bad_value: float, message: str
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "bad-error")
    errors = {key: dict(value) for key, value in _artifact_result().per_pair_errors.items()}
    errors[2]["left_rmse_px"] = bad_value
    result = replace(_artifact_result(), per_pair_errors=errors)

    with pytest.raises(ValueError, match=message):
        write_calibration_run(store, result, config, _source_pairs(store))

    assert list(store.results_dir.iterdir()) == []


def test_inconsistent_supplied_combined_error_is_rejected(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "bad-combined")
    errors = {key: dict(value) for key, value in _artifact_result().per_pair_errors.items()}
    errors[2]["combined_rmse_px"] = 0.9
    result = replace(_artifact_result(), per_pair_errors=errors)

    with pytest.raises(ValueError, match="combined_rmse_px is inconsistent"):
        write_calibration_run(store, result, config, _source_pairs(store))

    assert list(store.results_dir.iterdir()) == []


def test_map_shape_must_match_config_before_artifact_creation(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "bad-map")
    result = replace(
        _artifact_result(), map_right_y=np.zeros((47, 64), dtype=np.float32)
    )

    with pytest.raises(ValueError, match=r"map_right_y must have shape \(48, 64\)"):
        write_calibration_run(store, result, config, _source_pairs(store))

    assert list(store.results_dir.iterdir()) == []


def test_preview_source_dimensions_must_match_config_before_staging(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "bad-preview-size")
    sources = _source_pairs(store)
    wrong = np.zeros((48, 63, 3), dtype=np.uint8)
    assert cv2.imwrite(str(sources[1].left_path), wrong)
    assert cv2.imwrite(str(sources[1].right_path), wrong)

    with pytest.raises(ValueError, match="source pair 2 image dimensions 63x48"):
        write_calibration_run(store, _artifact_result(), config, sources)

    assert list(store.results_dir.iterdir()) == []


def test_outlier_is_reported_without_deleting_or_excluding_source_pair(
    tmp_path: Path,
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "outlier")
    sources = _source_pairs(store)
    result = _artifact_result(
        per_pair_errors={
            1: {"left_rmse_px": 0.1, "right_rmse_px": 0.1, "combined_rmse_px": 0.1},
            2: {"left_rmse_px": 0.2, "right_rmse_px": 0.2, "combined_rmse_px": 0.2},
            3: {"left_rmse_px": 4.0, "right_rmse_px": 4.0, "combined_rmse_px": 4.0},
        }
    )

    run_dir = write_calibration_run(store, result, config, sources)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert report["suspected_outliers"] == [
        {"pair_id": 3, "combined_reprojection_error_px": 4.0}
    ]
    assert report["source_pair_ids"] == [1, 2, 3]
    assert sources[2].left_path.is_file()
    assert sources[2].right_path.is_file()
    assert "3（组合重投影误差 4.000 px）" in (
        run_dir / "report.md"
    ).read_text(encoding="utf-8")


def test_baseline_warning_is_reported(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path, baseline_reference_mm=250.0)
    store = SessionStore.create(tmp_path, "baseline")
    run_dir = write_calibration_run(
        store, _artifact_result(), config, _source_pairs(store)
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert report["baseline_warning"]["warning"] is True
    assert report["baseline_warning"]["reference_mm"] == 250.0
    assert "基线警告：是" in (run_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"left_rms": 1.0, "right_rms": 1.0, "stereo_rms": 1.0}, "pass"),
        ({"left_rms": 1.0001}, "warning"),
        ({"stereo_rms": 1.5}, "warning"),
        ({"stereo_rms": 1.5001}, "fail"),
        ({"vertical_median": 1.0}, "warning"),
        ({"vertical_p95": 2.0}, "warning"),
    ],
)
def test_status_boundaries(
    tmp_path: Path, changes: dict[str, float], expected: str
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, f"status-{expected}-{len(list(tmp_path.iterdir()))}")
    result = _artifact_result(**changes)

    run_dir = write_calibration_run(store, result, config, _source_pairs(store))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert report["status"] == expected


def test_two_runs_use_distinct_directories_without_overwriting(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "twice")
    result = _artifact_result()
    sources = _source_pairs(store)

    first = write_calibration_run(store, result, config, sources)
    first_report = (first / "report.json").read_bytes()
    second = write_calibration_run(store, result, config, sources)

    assert first != second
    assert first_report == (first / "report.json").read_bytes()
    assert (second / "report.json").is_file()


def test_concurrent_writers_with_same_timestamp_publish_distinct_intact_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "concurrent-runs")
    second_store = SessionStore.open(store.session_dir)
    result = _artifact_result()
    sources = _source_pairs(store)
    frozen = datetime(2026, 8, 26, 12, 34, 56, 123456)
    monkeypatch.setattr(calibration, "_current_datetime", lambda: frozen)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_calibration_run, selected, result, config, sources)
            for selected in (store, second_store)
        ]
        run_dirs = [future.result() for future in futures]

    assert len(set(run_dirs)) == 2
    assert {path.name for path in run_dirs} == {
        "20260826-123456123456",
        "20260826-123456123457",
    }
    for path in run_dirs:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        assert report["source_pair_ids"] == [1, 2, 3]
        assert (path / "stereo_calibration.npz").is_file()


def test_reopen_validation_rejects_wrong_preview_shape_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "bad-reopen")
    sources = _source_pairs(store)
    original = calibration._write_rectified_previews

    def corrupt_preview(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        staging_dir = args[0]
        preview_path = sorted((staging_dir / "previews").glob("*.png"))[0]
        assert cv2.imwrite(str(preview_path), np.zeros((2, 2, 3), dtype=np.uint8))

    monkeypatch.setattr(calibration, "_write_rectified_previews", corrupt_preview)

    with pytest.raises(ValueError, match="preview.*unexpected shape"):
        write_calibration_run(store, _artifact_result(), config, sources)

    assert [path for path in store.results_dir.iterdir() if path.is_dir()] == []


def test_fewer_than_three_readable_pairs_fails_before_publishing(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "too-few")
    sources = _source_pairs(store)
    sources[2].right_path.unlink()

    with pytest.raises(ValueError, match="at least 3 readable source pairs"):
        write_calibration_run(store, _artifact_result(), config, sources)

    assert list(store.results_dir.iterdir()) == []


def test_artifact_failure_removes_staging_and_leaves_no_final_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "atomic")
    sources = _source_pairs(store)

    def fail_yaml(*args: object, **kwargs: object) -> None:
        raise OSError("injected YAML failure")

    monkeypatch.setattr(calibration, "_write_yaml_artifact", fail_yaml)
    with pytest.raises(OSError, match="injected YAML failure"):
        write_calibration_run(store, _artifact_result(), config, sources)

    assert not [path for path in store.results_dir.iterdir() if path.is_dir()]


def test_cleanup_failure_reports_primary_and_cleanup_and_stale_stage_is_not_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _artifact_config(tmp_path)
    store = SessionStore.create(tmp_path, "cleanup-failure")
    sources = _source_pairs(store)
    original_yaml = calibration._write_yaml_artifact
    original_cleanup = calibration._cleanup_staging

    def fail_yaml(*args: object, **kwargs: object) -> None:
        raise OSError("primary artifact failure")

    def fail_cleanup(path: Path) -> None:
        raise OSError("staging cleanup failure")

    monkeypatch.setattr(calibration, "_write_yaml_artifact", fail_yaml)
    monkeypatch.setattr(calibration, "_cleanup_staging", fail_cleanup)
    with pytest.raises(RuntimeError) as captured:
        write_calibration_run(store, _artifact_result(), config, sources)

    assert "primary artifact failure" in str(captured.value)
    assert "staging cleanup failure" in str(captured.value)
    assert isinstance(captured.value.__cause__, OSError)
    assert "primary artifact failure" in str(captured.value.__cause__)
    stale = [path for path in store.results_dir.iterdir() if path.is_dir()]
    assert len(stale) == 1
    assert stale[0].name.startswith(".") and stale[0].name.endswith(".tmp")

    monkeypatch.setattr(calibration, "_write_yaml_artifact", original_yaml)
    monkeypatch.setattr(calibration, "_cleanup_staging", original_cleanup)
    run_dir = write_calibration_run(store, _artifact_result(), config, sources)

    assert re.fullmatch(r"\d{8}-\d{12}", run_dir.name)
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["status"] == "pass"
    assert stale[0].is_dir()
