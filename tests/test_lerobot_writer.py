"""Unit tests for LeRobotDatasetWriter and RawRecorder."""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from Co_Teleop.recording.lerobot_writer import LeRobotDatasetWriter
from Co_Teleop.recording.raw_recorder import RawRecorder
from Co_Teleop.recording.teleop_frame import (
    ArmObservation,
    CameraFrame,
    FrameValidity,
    HandObservation,
    TaskMetadata,
    TeleopFrame,
)


def create_dummy_frame(validity: FrameValidity = FrameValidity.VALID, frame_id: int = 1) -> TeleopFrame:
    arm = ArmObservation(q=np.zeros(6), dq=np.zeros(6), current=np.zeros(6), timestamp=100.0)
    hand = HandObservation(q=np.zeros(16), currents=np.zeros(16), timestamp=100.0)
    cam = CameraFrame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=100.0)
    return TeleopFrame(
        frame_id=frame_id,
        timestamp=100.0,
        arm=arm,
        hand=hand,
        overhead_cam=cam,
        observation_state_22d=np.zeros(22, dtype=np.float32),
        action_22d=np.zeros(22, dtype=np.float32),
        cartesian_cmd=np.zeros(6, dtype=np.float32),
        validity=validity,
        task=TaskMetadata(task_name="按揉测试", target_acupoint="dazhui"),
    )


def test_lerobot_writer_and_cleaning():
    temp_dir = Path(tempfile.mkdtemp())
    ds_root = temp_dir / "tuina_writer_test"

    try:
        writer = LeRobotDatasetWriter(
            repo_id="test_writer_dataset",
            fps=30,
            root=ds_root,
        )
        assert writer.initialize() is True

        writer.start_episode()
        assert writer.is_recording is True

        # 1. Add valid frame -> should be accepted
        f1 = create_dummy_frame(FrameValidity.VALID, frame_id=1)
        assert writer.add_frame(f1) is True
        assert writer.episode_frames_count == 1

        # 2. Add PAUSED frame -> should be filtered out
        f_paused = create_dummy_frame(FrameValidity.PAUSED, frame_id=2)
        assert writer.add_frame(f_paused) is False
        assert writer.episode_frames_count == 1

        # 3. Add more valid frames
        f3 = create_dummy_frame(FrameValidity.VALID, frame_id=3)
        assert writer.add_frame(f3) is True
        assert writer.episode_frames_count == 2

        # 4. Save episode
        num_episodes = writer.save_episode()
        assert num_episodes == 1
        assert writer.is_recording is False

        # 5. Clear buffer on second episode
        writer.start_episode()
        writer.add_frame(f1)
        writer.clear_episode_buffer()
        assert writer.is_recording is False
        assert writer.episode_frames_count == 0

        writer.finalize()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_raw_recorder():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        raw_rec = RawRecorder(root=temp_dir)
        ep_dir = raw_rec.start_episode("ep_test_001")
        assert ep_dir.exists()
        assert raw_rec.is_recording is True

        f1 = create_dummy_frame(FrameValidity.VALID, frame_id=1)
        raw_rec.record_frame(f1)

        count = raw_rec.save_episode()
        assert count == 1
        assert (ep_dir / "raw_telemetry.jsonl").exists()

        # Test discard
        ep_dir2 = raw_rec.start_episode("ep_test_002")
        raw_rec.record_frame(f1)
        raw_rec.discard_episode()
        assert not ep_dir2.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
