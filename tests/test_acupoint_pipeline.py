"""Unit tests for AcuPointDet perception pipeline, worker, cache and dataset integration."""

import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from Co_Teleop.adapters.state_poller import LatestStateCache
from Co_Teleop.perception.acupoint.mock_backend import MockAcupointBackend
from Co_Teleop.perception.acupoint.observation import AcupointObservation
from Co_Teleop.perception.acupoint.worker import AcuPointWorker
from Co_Teleop.perception.acupoint_adapter import create_acupoint_detector
from Co_Teleop.pipeline.observation_aggregator import ObservationAggregator
from Co_Teleop.recording.lerobot_writer import LeRobotDatasetWriter
from Co_Teleop.recording.teleop_frame import (
    ArmObservation,
    CameraFrame,
    FrameValidity,
    HandObservation,
    TaskMetadata,
    TeleopFrame,
)


def test_mock_acupoint_backend():
    detector = MockAcupointBackend(default_confidence=0.95)
    assert detector.initialize() is True

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = 100.0
    obs = detector.detect(img, timestamp=t0, target_acupoint="dazhui")

    assert obs is not None
    assert isinstance(obs, AcupointObservation)
    assert obs.keypoints_2d.shape == (37, 3)
    assert obs.bbox.shape == (4,)
    assert obs.confidence == 0.95
    assert obs.target_acupoint == "dazhui"
    assert obs.is_valid is True

    detector.close()


def test_acupoint_adapter_factory():
    # Factory with mock
    det_mock = create_acupoint_detector(backend_type="mock")
    assert isinstance(det_mock, MockAcupointBackend)
    assert det_mock.initialize() is True

    # Factory with auto (in dev environment without Ascend, falls back to Mock)
    det_auto = create_acupoint_detector(backend_type="auto")
    assert det_auto is not None
    assert det_auto.initialize() is True


def test_acupoint_worker_async_loop():
    cache = LatestStateCache()
    detector = MockAcupointBackend()
    detector.initialize()

    worker = AcuPointWorker(detector=detector, cache=cache, max_rate_hz=50.0)
    worker.start()

    # Submit test frame
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    worker.submit_frame(img, timestamp=time.monotonic())

    # Wait briefly for worker to process
    time.sleep(0.08)

    snap = cache.snapshot()
    assert snap.acupoints is not None
    assert snap.acupoints.keypoints_2d.shape == (37, 3)
    assert snap.acupoints.is_valid is True

    worker.stop()


def test_observation_aggregator_with_acupoints():
    cache = LatestStateCache()
    aggregator = ObservationAggregator(cache)

    t0 = 100.0
    kpts = np.ones((37, 3), dtype=np.float32)
    bbox = np.array([10.0, 20.0, 100.0, 200.0], dtype=np.float32)
    acu_obs = AcupointObservation(
        keypoints_2d=kpts,
        bbox=bbox,
        confidence=0.9,
        timestamp=t0,
        target_acupoint="jianjing_L",
    )
    cache.update_acupoints(acu_obs)

    action_22d = np.zeros(22, dtype=np.float32)
    cart_cmd = np.zeros(6, dtype=np.float32)
    frame = aggregator.aggregate(action_22d, cart_cmd, now=t0 + 0.01)

    assert frame.acupoints is not None
    assert frame.acupoints.keypoints_2d.shape == (37, 3)
    assert frame.acupoints.target_acupoint == "jianjing_L"

    # Check to_lerobot_dict contains acupoints
    lr_dict = frame.to_lerobot_dict()
    assert "observation.environment.acupoints" in lr_dict
    assert lr_dict["observation.environment.acupoints"].shape == (37, 3)


def test_lerobot_writer_acupoint_dataset_persistence():
    try:
        import lerobot
    except ImportError:
        pytest.skip("lerobot 库未安装于当前 Python 环境，跳过磁盘数据集持久化测试")

    temp_dir = Path(tempfile.mkdtemp())
    ds_root = temp_dir / "lerobot_acu_ds"

    try:
        writer = LeRobotDatasetWriter(repo_id="test_acu_ds", fps=30, root=ds_root)
        assert writer.initialize() is True

        writer.start_episode()
        for i in range(5):
            arm = ArmObservation(q=np.zeros(6), dq=np.zeros(6), current=np.zeros(6), timestamp=100.0 + i * 0.033)
            hand = HandObservation(q=np.zeros(16), currents=np.zeros(16), timestamp=100.0 + i * 0.033)
            cam = CameraFrame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=100.0 + i * 0.033)
            acu = AcupointObservation(
                keypoints_2d=np.full((37, 3), i * 0.1, dtype=np.float32),
                bbox=np.array([0, 0, 100, 100], dtype=np.float32),
                confidence=0.95,
                timestamp=100.0 + i * 0.033,
            )
            frame = TeleopFrame(
                frame_id=i + 1,
                timestamp=100.0 + i * 0.033,
                arm=arm,
                hand=hand,
                overhead_cam=cam,
                observation_state_22d=np.zeros(22, dtype=np.float32),
                action_22d=np.zeros(22, dtype=np.float32),
                cartesian_cmd=np.zeros(6, dtype=np.float32),
                acupoints=acu,
                validity=FrameValidity.VALID,
                task=TaskMetadata(task_name="大椎穴按揉", target_acupoint="dazhui"),
            )
            saved = writer.add_frame(frame)
            assert saved is True

        num_eps = writer.save_episode()
        assert num_eps == 1
        writer.finalize()

        assert (ds_root / "data").exists()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
