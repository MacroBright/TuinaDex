"""Unit tests for dataset replay and tracking error calculations."""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from tools.dataset_replay import calculate_tracking_metrics, replay_episode
from Co_Teleop.recording.lerobot_writer import LeRobotDatasetWriter
from Co_Teleop.recording.teleop_frame import (
    ArmObservation,
    CameraFrame,
    FrameValidity,
    HandObservation,
    TaskMetadata,
    TeleopFrame,
)


def test_tracking_metrics_calculation():
    # Commanded 22D constant, executed with known error
    commanded = np.zeros((100, 22), dtype=np.float32)
    executed = np.zeros((100, 22), dtype=np.float32)
    executed[:, 0] = 0.05  # 0.05 rad error on J1

    metrics = calculate_tracking_metrics(commanded, executed)
    assert "rmse" in metrics
    assert "max_error" in metrics
    assert "p95_error" in metrics

    assert np.isclose(metrics["rmse"][0], 0.05, atol=1e-4)
    assert np.isclose(metrics["max_error"][0], 0.05, atol=1e-4)
    assert np.isclose(metrics["p95_error"][0], 0.05, atol=1e-4)
    assert np.isclose(metrics["rmse"][1], 0.0, atol=1e-4)


def test_replay_dry_run_and_sim():
    temp_dir = Path(tempfile.mkdtemp())
    ds_root = temp_dir / "replay_test_dataset"

    try:
        # Create small 1-episode dataset
        writer = LeRobotDatasetWriter(repo_id="test_replay_ds", fps=30, root=ds_root)
        assert writer.initialize() is True

        writer.start_episode()
        for i in range(10):
            arm = ArmObservation(q=np.zeros(6), dq=np.zeros(6), current=np.zeros(6), timestamp=100.0 + i * 0.033)
            hand = HandObservation(q=np.zeros(16), currents=np.zeros(16), timestamp=100.0 + i * 0.033)
            cam = CameraFrame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=100.0 + i * 0.033)
            frame = TeleopFrame(
                frame_id=i + 1,
                timestamp=100.0 + i * 0.033,
                arm=arm,
                hand=hand,
                overhead_cam=cam,
                observation_state_22d=np.zeros(22, dtype=np.float32),
                action_22d=np.zeros(22, dtype=np.float32),
                cartesian_cmd=np.zeros(6, dtype=np.float32),
                validity=FrameValidity.VALID,
                task=TaskMetadata(task_name="重放测试", target_acupoint="dazhui"),
            )
            writer.add_frame(frame)
        writer.save_episode()
        writer.finalize()

        # 1. Test Dry-Run
        dry_ok = replay_episode(dataset_path=ds_root, episode_idx=0, mode="dry-run")
        assert dry_ok is True

        # 2. Test Sim replay
        sim_ok = replay_episode(dataset_path=ds_root, episode_idx=0, mode="sim", speed_ratio=1.0)
        assert sim_ok is True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
